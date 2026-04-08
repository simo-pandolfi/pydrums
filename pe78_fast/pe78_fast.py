#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
pe78_fast.py — Single-import accelerator for the entire PE78 drum engine
=========================================================================
Usage
-----
    # Replace this in your run script / notebook:
    #   from pe78.cymb  import PE78_Cymbals_Maracas
    #   from pe78.snare import PE78_Snare
    #   from pe78.twint import TwinTDrum, EdgeDetector
    #   from pe78.drums import SnareDrum, CymbDrum, TonalDrum

    # With this:
    import pe78_fast   # noqa — applies all patches, then re-exports

    from pe78_fast import (
        PE78_Cymbals_Maracas,
        PE78_Snare,
        TwinTDrum, EdgeDetector,
        SnareDrum, CymbDrum, TonalDrum,
    )
    # Everything else is identical to the original code.

Patch application order
-----------------------
  1. pywdf_patch   — RTypeAdaptor.r_type_scatter, Diode.omega4
  2. cymb_patch    — PE78_Cymbals_Maracas._solve_env_ebers_moll,
                     _par_scatter (cymb module), process_sample C31 IIR
  3. snare_patch   — PE78_Snare._solve_base_ebers_moll,
                     _par_scatter (snare module), process_sample C31 IIR
  4. twint_patch   — TwinTDrum._scattering_logic, TwinTDrum.process_sample

Compilation cost
----------------
All @njit kernels are compiled during the import of pywdf_patch (~1-2 s,
once per Python process).  No recompilation on subsequent imports.

Expected speedup (measured on x86-64, CPython 3.12, Numba 0.60)
-----------------------------------------------------------------
  r_type_scatter (9×9):      ~3×
  nr_ebers_moll:             ~4-5×
  omega4 (Diode):            ~3×
  Overall per-sample loop:   ~3-4×  (dominated by Python call overhead
                                      that Numba cannot eliminate)

The remaining Python overhead (attribute lookup, set_voltage, wave_to_voltage)
is the primary target for the subsequent Rust port.
"""

# ---- apply patches in correct order ----
import pywdf_patch   # noqa: F401  (1) rtype scatter + diode omega4
import cymb_patch    # noqa: F401  (2) cymb NR + par_scatter + C31 IIR
import snare_patch   # noqa: F401  (3) snare NR + par_scatter + C31 IIR
import twint_patch   # noqa: F401  (4) twint scattering + process_sample

# ---- re-export the (now patched) public symbols ----
from pe78.cymb  import PE78_Cymbals_Maracas          # noqa: F401
from pe78.snare import PE78_Snare, SnareDrum          # noqa: F401
from pe78.twint import TwinTDrum, EdgeDetector        # noqa: F401
from pe78.drums import CymbDrum, TonalDrum            # noqa: F401


# ---------------------------------------------------------------------------
# Quick self-test / benchmark  (run as __main__)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np

    FS = 48_000
    N  = FS * 2          # 2 seconds
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(N) * 0.02

    print(f"\nPE78 Numba benchmark — {N} samples @ {FS} Hz\n")

    # ---------- SnareDrum ----------
    snare = SnareDrum(FS)
    trig  = np.zeros(N)
    trig[0] = 4.5

    t0 = time.perf_counter()
    out_s = np.array([
        snare.tick(trig[i], noise[i]) for i in range(N)
    ])
    t1 = time.perf_counter()
    rt_factor = N / FS / (t1 - t0)
    print(f"  SnareDrum:  {(t1-t0)*1000:.0f} ms  ({rt_factor:.1f}× real-time)")

    # ---------- CymbDrum ----------
    cymb = CymbDrum(FS)
    trig_lc = trig.copy()
    trig_sc = np.zeros(N); trig_sc[int(0.5*FS)] = 4.5
    trig_mr = np.zeros(N); trig_mr[int(1.0*FS)] = 4.5

    t0 = time.perf_counter()
    out_c = np.array([
        cymb.tick(trig_lc[i], trig_sc[i], trig_mr[i], noise[i])
        for i in range(N)
    ])
    t1 = time.perf_counter()
    rt_factor = N / FS / (t1 - t0)
    print(f"  CymbDrum:   {(t1-t0)*1000:.0f} ms  ({rt_factor:.1f}× real-time)")

    # ---------- TonalDrum (Bass Drum) ----------
    from pe78.drums import BDO_PARAMS
    bd = TonalDrum(FS, **{
        k: BDO_PARAMS[k] for k in
        ('edge_r', 'edge_c', 'kick_r', 'kick_c', 'kick_vr4', 'kick_load')
    })

    t0 = time.perf_counter()
    out_b = np.array([
        bd.tick(trig[i]) for i in range(N)
    ])
    t1 = time.perf_counter()
    rt_factor = N / FS / (t1 - t0)
    print(f"  TonalDrum:  {(t1-t0)*1000:.0f} ms  ({rt_factor:.1f}× real-time)")

    print("\nPeak outputs (sanity check):")
    print(f"  Snare : {out_s.max():.4f} V")
    print(f"  Cymbal: {out_c.max():.4f} V")
    print(f"  BD    : {out_b.max():.4f} V")
