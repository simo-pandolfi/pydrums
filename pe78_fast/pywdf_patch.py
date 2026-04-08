#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
pywdf_patch.py — Monkey-patch pywdf with Numba JIT kernels
===========================================================
Apply at startup (before instantiating any circuit):

    import pywdf_patch   # noqa: F401

What is patched
---------------
  RTypeAdaptor.r_type_scatter()  → wdf_kernels.r_type_scatter()
  Diode.omega4()                 → wdf_kernels.omega4()   (staticmethod)
  DiodePair inherits the patched omega4 automatically.

Both patches are backward-compatible: the Python interface and attribute
layout are unchanged; only the inner computation is accelerated.

Compilation cost: ~1-2 s on first import (JIT warmup), zero thereafter.

Usage example
-------------
    # At the top of your run script / notebook:
    import pywdf_patch        # triggers JIT compilation once

    from pe78.cymb  import PE78_Cymbals_Maracas
    from pe78.snare import PE78_Snare
    from pe78.twint import TwinTDrum, EdgeDetector
    # ... rest of your code unchanged ...
"""

import numpy as np
import pywdf.core.rtype as _rtype
import pywdf.core.wdf   as _wdf
from pe78_fast.wdf_kernels import r_type_scatter as _jit_scatter, omega4 as _jit_omega4


# ---------------------------------------------------------------------------
# Patch 1 — RTypeAdaptor.r_type_scatter
# ---------------------------------------------------------------------------
# Original (pywdf/core/rtype.py):
#
#   def r_type_scatter(self) -> None:
#       for i in range(self.n_ports):
#           self.b_vals[i] = 0
#           for j in range(self.n_ports):
#               self.b_vals[i] += self.S_matrix[i][j] * self.a_vals[j]
#
# Replacement: delegates to @njit kernel, in-place, no allocation.

def _r_type_scatter_fast(self) -> None:
    _jit_scatter(self.S_matrix, self.a_vals, self.b_vals)

_rtype.RTypeAdaptor.r_type_scatter = _r_type_scatter_fast


# ---------------------------------------------------------------------------
# Patch 2 — Diode.omega4  (and DiodePair via inheritance)
# ---------------------------------------------------------------------------
# Original is a regular Python method; replace with a staticmethod wrapper
# that calls the @njit version.
#
# Note: omega4 is called as self.omega4(x) in propagate_reflected_wave().
# The lambda wrapper preserves this calling convention.

def _omega4_fast(self, x: float) -> float:   # self ignored — pure function
    return _jit_omega4(x)

_wdf.Diode.omega4 = _omega4_fast


# ---------------------------------------------------------------------------
# JIT warmup — compile all kernels now (pays the one-time cost at import)
# ---------------------------------------------------------------------------
def _warmup():
    from pe78_fast.wdf_kernels import (
        r_type_scatter, omega4, nr_ebers_moll,
        bilinear_hp_step, iir_scalar,
        par_scatter_4, par_scatter_5,
    )
    # Trigger compilation with realistic argument types
    S9  = np.eye(9, dtype=np.float64)
    a9  = np.zeros(9, dtype=np.float64)
    b9  = np.zeros(9, dtype=np.float64)
    r_type_scatter(S9, a9, b9)

    S4  = np.eye(4, dtype=np.float64)
    G4  = np.ones(4, dtype=np.float64)
    par_scatter_4(G4, 4.0, S4)

    S5  = np.eye(5, dtype=np.float64)
    G5  = np.ones(5, dtype=np.float64)
    par_scatter_5(G5, 5.0, S5)

    nr_ebers_moll(2e-6, -1e-6, 0.6, 7e-17, 26e-3)
    omega4(0.5)
    bilinear_hp_step(1.0, 0.0, 0.0, 0.9)
    iir_scalar(1.0, 0.0, 0.0, 0.5, -0.5, -0.9)

_warmup()
print("[pywdf_patch] Numba JIT kernels compiled and active.")
