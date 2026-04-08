#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
cymb_patch.py — Numba acceleration patches for PE78_Cymbals_Maracas
====================================================================
Apply AFTER importing cymb, BEFORE instantiating PE78_Cymbals_Maracas:

    import pywdf_patch        # patches rtype + diode (must come first)
    import cymb_patch         # patches cymb-specific hot paths
    from pe78.cymb import PE78_Cymbals_Maracas

Patched methods
---------------
  _par_scatter()                       module-level helper  → @njit kernel
  PE78_Cymbals_Maracas._solve_env_ebers_moll()  → @njit nr_ebers_moll
  PE78_Cymbals_Maracas.process_sample()         C31 IIR step → @njit iir_scalar

The patches do NOT change any attribute or public interface.

Implementation note: _par_scatter is a module-level function referenced by
  RootRTypeAdaptor constructors.  Since pywdf stores the callable reference
  at construction time (adaptor.impedance_calc = _par_scatter), instances
  already built before this patch is applied will NOT be affected.
  Always apply the patch before constructing PE78_Cymbals_Maracas.
"""

import numpy as np
import pe78.cymb as _cymb
from pe78.cymb import PE78_Cymbals_Maracas
from pe78_fast.wdf_kernels import (
    nr_ebers_moll  as _nr_jit,
    par_scatter_n  as _par_n_jit,
    iir_scalar     as _iir_jit,
)


# ---------------------------------------------------------------------------
# Patch 1 — _par_scatter (module-level, must replace before construction)
# ---------------------------------------------------------------------------
# Original (cymb.py):
#
#   def _par_scatter(adaptor):
#       G  = [1.0 / p.Rp for p in adaptor.down_ports]
#       Gt = sum(G)
#       n  = len(G)
#       S  = np.zeros((n, n))
#       for i in range(n): for j in range(n):
#           S[i][j] = 2.0 * G[j] / Gt - (1.0 if i == j else 0.0)
#       adaptor.set_S_matrix(S)
#       adaptor.Rp = 1.0 / Gt
#
# Replacement: reuses a pre-allocated buffer; calls @njit par_scatter_n.
# The buffer is stored on the adaptor itself after the first call (duck-typing).

def _par_scatter_fast(adaptor):
    ports = adaptor.down_ports
    n     = len(ports)
    # Reuse buffer if already allocated on this adaptor
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

_cymb._par_scatter = _par_scatter_fast   # module-level replacement


# ---------------------------------------------------------------------------
# Patch 2 — PE78_Cymbals_Maracas._solve_env_ebers_moll
# ---------------------------------------------------------------------------
# The original method builds G_fixed / I_fixed by iterating over down_ports,
# then runs a 20-iteration NR loop in pure Python.  Replaced by @njit kernel.

def _solve_env_ebers_moll_fast(self):
    """Accelerated Ebers-Moll NR for ENV node (cymb.py)."""
    a_vals  = self._root_env.a_vals
    ports   = self._root_env.down_ports
    idx_rpi = self._RPI_PORT_IDX

    G_fixed = 0.0
    I_fixed = 0.0
    for k, p in enumerate(ports):
        if k == idx_rpi:
            continue
        gk       = 1.0 / p.Rp
        G_fixed += gk
        I_fixed += gk * a_vals[k]

    V = _nr_jit(G_fixed, I_fixed, self._V_env, self.Is_b, self.Vt)

    # Collector operating point
    arg    = np.clip(V / self.Vt, -40.0, 40.0)
    ex_sol = np.exp(arg)
    Ic     = self.hFE * self.Is_b * ex_sol
    Ic     = min(Ic, (self.Vcc - 0.2) / self.R54)
    Ic     = max(Ic, 1e-15)
    gm     = Ic / self.Vt
    r_pi   = float(np.clip(self.hFE * self.Vt / Ic,
                            self.R_PI_MIN, self.R_PI_MAX))
    return V, gm, r_pi

PE78_Cymbals_Maracas._solve_env_ebers_moll = _solve_env_ebers_moll_fast


# ---------------------------------------------------------------------------
# Patch 3 — C31 IIR step inside process_sample
# ---------------------------------------------------------------------------
# The C31 IIR in process_sample() is three lines of Python arithmetic.
# We wrap only that step; the rest of process_sample is unchanged.
# Strategy: override the whole method minimally by injecting iir_scalar.

_original_process_sample = PE78_Cymbals_Maracas.process_sample

def _process_sample_fast(self, v_trig_lc, v_trig_sc, v_trig_mr,
                          noise_sample=0.0):
    # Run all existing logic up to C31 by calling the original; we re-do
    # only the IIR step, so we temporarily swap the IIR internals.
    # Simpler: just inline the IIR fix here directly.
    # Since the original already returns y, we patch the IIR call inside:

    # ---- replicate original body, replacing only the C31 IIR ----
    on_lc = v_trig_lc > 0.5
    on_sc = v_trig_sc > 0.5
    on_mr = v_trig_mr > 0.5

    if on_lc != self._on_lc:
        self._rvsrc_lc.set_resistance(self.R_ON if on_lc else self.R_OFF)
        self._on_lc = on_lc
    if on_sc != self._on_sc:
        self._rvsrc_sc.set_resistance(self.R_ON if on_sc else self.R_OFF)
        self._on_sc = on_sc
    if on_mr != self._on_mr:
        self._rvsrc_mr.set_resistance(self.R_ON if on_mr else self.R_OFF)
        self._on_mr = on_mr

    self._rvsrc_lc.set_voltage(self.V_ON if on_lc else 0.0)
    self._rvsrc_sc.set_voltage(-(self.V_ON if on_sc else 0.0))
    self._rvsrc_mr.set_voltage(self.V_ON if on_mr else 0.0)

    R_src_mr    = self._rvsrc_mr.Rval
    V_src_mr    = self.V_ON if on_mr else 0.0
    R_par_mr    = R_src_mr * self.R43 / (R_src_mr + self.R43)
    V_anode_tgt = V_src_mr * self.R43 / (R_src_mr + self.R43)
    tau_c25     = self.C25 * R_par_mr
    self._V_c25 += self._dt / (tau_c25 + self._dt) * (V_anode_tgt - self._V_c25)

    self._rvsrc_anode_th.set_resistance(R_par_mr)
    self._rvsrc_anode_th.set_voltage(-self._V_c25)

    self._rvsrc_mr_port.set_resistance(self.R46)
    self._rvsrc_mr_port.set_voltage(-self._V_c26)

    R_src_lc = self._rvsrc_lc.Rval
    R_src_sc = self._rvsrc_sc.Rval
    R_env_th = 1.0 / (
        1.0 / (self.R47 + R_src_lc)
        + 1.0 / (self.R42 + self.R44 + R_src_sc)
        + 1.0 / self.R52
        + 1.0 / self._r_pi_applied
    )
    self._rvsrc_env_th.set_resistance(R_env_th)
    self._rvsrc_env_th.set_voltage(self._V_env)

    self._d9.accept_incident_wave(self._tree_mr.propagate_reflected_wave())
    self._tree_mr.accept_incident_wave(self._d9.propagate_reflected_wave())
    self._V_c26 = -self._c26.wave_to_voltage()

    self._root_env.compute()

    V_env_new, gm_new, r_pi_new = self._solve_env_ebers_moll()
    self._V_env = V_env_new
    self._V_c23 = self._c23.wave_to_voltage()
    self._V_c24 = self._c24.wave_to_voltage()

    if abs(r_pi_new - self._r_pi_applied) / self._r_pi_applied > 1e-4:
        self._r_pi_port.set_resistance(r_pi_new)
        self._r_pi_applied = r_pi_new

    V_col       = self._step_tr4(V_env_new, gm_new, noise_sample * self.NOISE_AMP)
    self._V_col = V_col

    # ---- C31 IIR — @njit scalar step ----
    y = _iir_jit(V_col, self._c31_x, self._c31_y,
                 self._b_c31[0], self._b_c31[1], self._a_c31[1])
    self._c31_x = V_col
    self._c31_y = y
    return y

PE78_Cymbals_Maracas.process_sample = _process_sample_fast
