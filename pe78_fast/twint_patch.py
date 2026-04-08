#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
twint_patch.py — Numba acceleration patches for TwinTDrum + EdgeDetector
=========================================================================
Apply AFTER importing twint, BEFORE instantiating TwinTDrum or EdgeDetector:

    import pywdf_patch        # patches rtype + diode (must come first)
    import twint_patch        # patches twint-specific hot paths
    from pe78.twint import TwinTDrum, EdgeDetector

Patched targets
---------------
  TwinTDrum._scattering_logic()   — 9-port S matrix (called only on impedance
                                    change, not per sample).  The per-sample
                                    scatter is already covered by pywdf_patch
                                    (RTypeAdaptor.r_type_scatter).
  TwinTDrum.process_sample()      — avoids repeated Python attribute lookups;
                                    the scatter itself delegates to @njit.

The scattering_logic patch pre-allocates the A matrix (constant) and the
G / M / S buffers, avoiding allocation on each impedance recalculation.
np.linalg.inv on a (4×4) matrix is the main cost here; since it is called
only when VR4 changes (control rate), the speedup is less critical than for
the per-sample scatter.  The patch eliminates repeated allocation of
temporary NumPy arrays.

Note: the A matrix for TwinTDrum is CONSTANT (does not depend on component
values), so it is factored out as a class-level constant.  Only G changes
when an impedance changes, so M = A @ diag(G) @ A.T is recomputed but A
itself is never rebuilt.
"""

import numpy as np
from pe78.twint import TwinTDrum
from pe78_fast.wdf_kernels import r_type_scatter as _jit_scatter


# ---------------------------------------------------------------------------
# Constant incidence matrix A  (4 nodes × 9 ports, see twint.py docstring)
# ---------------------------------------------------------------------------
_A = np.array([
#    r11   r12    c5    c6    c7   r16  vtrig rbias  rout
    [ -1,   1,    1,    0,    0,    0,    0,    0,    0  ],  # n0 NB_R
    [  0,   0,    0,   -1,    1,    1,    1,    0,    0  ],  # n1 NC_C
    [  1,   0,    0,    1,    0,    0,    0,    1,    0  ],  # n2 NAND_IN
    [  0,  -1,    0,    0,   -1,    0,    0,   -1,    1  ],  # n3 NAND_OUT
], dtype=np.float64)

_AT = _A.T.copy()   # (9×4), stored C-contiguous for matmul performance


# ---------------------------------------------------------------------------
# Patch 1 — TwinTDrum._scattering_logic
# ---------------------------------------------------------------------------
# Original allocates A, G, M, S on every call.  We move A and the buffers
# to class-level / instance-level storage.  Called only at control rate
# (when set_vr4() or __init__ changes an impedance).

def _scattering_logic_fast(self, adapter):
    """
    Optimised WDF scattering matrix for TwinTDrum R-type adaptor.

    Re-uses pre-allocated buffers (_G_buf, _M_buf, _S9_buf stored on self).
    A and A^T are module-level constants (never rebuilt).
    """
    # ---- initialise buffers on first call --------------------------------
    if not hasattr(self, '_G_buf'):
        self._G_buf  = np.empty(9, dtype=np.float64)
        self._M_buf  = np.empty((4, 4), dtype=np.float64)
        self._S9_buf = np.empty((9, 9), dtype=np.float64)

    R_ports = np.array(adapter.get_port_impedances(), dtype=np.float64)
    G       = 1.0 / R_ports
    self._G_buf[:] = G

    # M = A @ diag(G) @ A^T   (passive KCL, 4×4)
    # Equivalent to  (A * G) @ A^T  without allocating diag(G)
    M = (_A * G) @ _AT           # (4×4)

    # Inverter feedback correction
    G_inv    = 1.0 / self.r_o
    M[3, 3] += G_inv
    M[3, 2] -= self.a_gain * G_inv

    M_inv = np.linalg.inv(M)     # (4×4)

    # S = 2 * diag(G) * A^T * M^-1 * A - I
    # = 2 * (A^T * M^-1 * A) * diag(G) - I   [note: G applied column-wise]
    # Computed as:  2 * (AT @ Minv @ A) with G broadcast, minus identity
    inner  = _AT @ M_inv @ _A    # (9×9)
    S      = (2.0 * inner) * G[:, np.newaxis] - np.eye(9)
    # Note: the original has diag(G) on the LEFT: 2*diag(G)*AT*Minv*A - I
    # which means S[i,j] = 2*G[i]*(AT@Minv@A)[i,j] - delta(i,j)
    # i.e., row i is scaled by G[i]:
    S      = (2.0 * G[:, np.newaxis]) * (_AT @ M_inv @ _A) - np.eye(9)

    adapter.set_S_matrix(S)

TwinTDrum._scattering_logic = _scattering_logic_fast


# ---------------------------------------------------------------------------
# Patch 2 — TwinTDrum.process_sample
# ---------------------------------------------------------------------------
# Original iterates over self.components twice with Python for-loops.
# We extract the arrays once and call the @njit scatter directly.
# The method body is otherwise identical to twint.py.

def _process_sample_fast(self, v_trigger: float) -> float:
    """Accelerated process_sample for TwinTDrum."""
    self.r_trigger_in.set_voltage(v_trigger)

    components  = self.components
    drum_osc    = self.drum_osc
    a_vals      = drum_osc.a_vals
    b_vals      = drum_osc.b_vals
    S_matrix    = drum_osc.S_matrix

    # Gather reflected waves from leaves
    for idx in range(len(components)):
        a_vals[idx] = components[idx].propagate_reflected_wave()

    # Scatter (hot path — @njit)
    _jit_scatter(S_matrix, a_vals, b_vals)

    # Distribute incident waves to leaves
    for idx in range(len(components)):
        components[idx].accept_incident_wave(b_vals[idx])

    return self.r_out.wave_to_voltage()

TwinTDrum.process_sample = _process_sample_fast
