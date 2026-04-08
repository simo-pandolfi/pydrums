#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
snare_patch.py — Numba acceleration patches for PE78_Snare
===========================================================
Apply AFTER importing snare, BEFORE instantiating PE78_Snare:

    import pywdf_patch        # patches rtype + diode (must come first)
    import snare_patch        # patches snare-specific hot paths
    from pe78.snare import PE78_Snare

Patched methods
---------------
  _par_scatter()                       module-level helper  → @njit kernel
  PE78_Snare._solve_base_ebers_moll()  → @njit nr_ebers_moll
  PE78_Snare.process_sample()          C31 IIR step         → @njit iir_scalar

The patches do NOT change any attribute or public interface.

See cymb_patch.py for the design rationale.
"""

import numpy as np
import pe78.snare as _snare
from pe78.snare import PE78_Snare
from pe78_fast.wdf_kernels import (
    nr_ebers_moll  as _nr_jit,
    par_scatter_n  as _par_n_jit,
    iir_scalar     as _iir_jit,
)


# ---------------------------------------------------------------------------
# Patch 1 — _par_scatter (module-level, must replace before construction)
# ---------------------------------------------------------------------------
# Identical strategy to cymb_patch.py: buffer stored on adaptor, @njit fill.

def _par_scatter_fast(adaptor):
    ports = adaptor.down_ports
    n     = len(ports)
    if not hasattr(adaptor, '_G_buf') or adaptor._G_buf.shape[0] != n:
        adaptor._G_buf = np.empty(n, dtype=np.float64)
        adaptor._S_buf = np.zeros((n, n), dtype=np.float64)
    G = adaptor._G_buf
    for k in range(n):
        G[k] = 1.0 / ports[k].Rp
    Gt = G.sum()
    _par_n_jit(G, Gt, adaptor._S_buf)
    adaptor.set_S_matrix(adaptor._S_buf)
    adaptor.Rp = 1.0 / Gt

_snare._par_scatter = _par_scatter_fast


# ---------------------------------------------------------------------------
# Patch 2 — PE78_Snare._solve_base_ebers_moll
# ---------------------------------------------------------------------------
# Structurally identical to cymb_patch but with the two-resistor Thevenin
# (R50, R51) instead of the multi-branch ENV junction.

def _solve_base_ebers_moll_fast(self, V_env: float):
    """Accelerated Ebers-Moll NR for TR3 base node (snare.py)."""
    G_fixed = 1.0 / self.R50 + 1.0 / self.R51
    I_fixed = -V_env / self.R50          # envelope pushes current in

    V = _nr_jit(G_fixed, I_fixed, self._V_base, self.Is_b, self.Vt)

    arg    = np.clip(V / self.Vt, -40.0, 40.0)
    ex_sol = np.exp(arg)
    Ic     = self.hFE * self.Is_b * ex_sol
    Ic     = min(Ic, (self.Vcc - 0.2) / self.R53)
    Ic     = max(Ic, 1e-15)
    gm     = Ic / self.Vt
    r_pi   = float(np.clip(self.hFE * self.Vt / Ic,
                            self.R_PI_MIN, self.R_PI_MAX))
    return V, gm, r_pi

PE78_Snare._solve_base_ebers_moll = _solve_base_ebers_moll_fast


# ---------------------------------------------------------------------------
# Patch 3 — C31 IIR step inside PE78_Snare.process_sample
# ---------------------------------------------------------------------------
# We patch the full method to replace only the final IIR arithmetic.
# The rest of the body is reproduced verbatim from snare.py.

def _process_sample_fast(self, v_trigger: float,
                          noise_sample: float = None) -> float:
    """Accelerated process_sample for PE78_Snare."""
    if noise_sample is None:
        noise_sample = np.random.normal(0.0, 1.0)

    # ---- sub-circuit A: envelope (IIR scalar) ----------------------------
    V_env = self._step_envelope(v_trigger)

    # ---- sub-circuit B: TR3 base NR (now @njit) --------------------------
    V_base, gm, r_pi = self._solve_base_ebers_moll(V_env)
    self._V_base = V_base

    # ---- noise coupling C28 ----------------------------------------------
    v_b = self._step_c28(noise_sample * self.NOISE_AMP, gm)

    # ---- sub-circuit C: TR3 collector WDF --------------------------------
    V_col       = self._step_tr3_collector(gm, v_b)
    self._V_col = V_col

    # ---- C31 IIR HP — @njit scalar step ----------------------------------
    y = _iir_jit(V_col, self._c31_x, self._c31_y,
                 self._b_c31[0], self._b_c31[1], self._a_c31[1])
    self._c31_x = V_col
    self._c31_y = y
    return y

PE78_Snare.process_sample = _process_sample_fast
