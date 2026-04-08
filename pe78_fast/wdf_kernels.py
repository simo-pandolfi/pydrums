#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
wdf_kernels.py — Numba JIT kernels for PE78 WDF hot paths
==========================================================
Drop-in acceleration for pywdf + PE78_Snare + PE78_Cymbals_Maracas.

All functions are @njit (no-Python, AOT-friendly).  The first call of each
kernel triggers JIT compilation (~0.5-2 s total); subsequent calls are fast.

Exported symbols
----------------
  r_type_scatter(S, a_vals, b_vals)   — RTypeAdaptor inner loop (replaces
                                         numpy matrix-vector product)
  par_scatter_4(G, Gt) -> S           — 4-port star-junction S matrix
  par_scatter_5(G, Gt) -> S           — 5-port star-junction S matrix
  nr_ebers_moll(G_fixed, I_fixed,     — Newton-Raphson BJT base/ENV solver
                V0, Is_b, Vt,           (cymb.py + snare.py)
                max_iter, tol)
  omega4(x)                           — Wright Omega 4th-order approx
                                         (wdf.Diode kernel)
  bilinear_hp(x, x_prev, y_prev,      — 1st-order bilinear HP IIR step
              alpha)                     (C28/C29, C31)
  iir_scalar(x, x_prev, y_prev,       — Direct-form-I scalar 1st-order IIR
             b0, b1, a1)                (C31 output HP)

Migration guide
---------------
1. Copy this file next to pywdf/core/ (or anywhere on sys.path).
2. Patch the three hot spots as shown in the companion patch files:
     - pywdf_patch.py   → patches RTypeAdaptor.r_type_scatter()
     - cymb_patch.py    → patches PE78_Cymbals_Maracas._solve_env_ebers_moll()
                           and _par_scatter()
     - snare_patch.py   → patches PE78_Snare._solve_base_ebers_moll()
                           and _par_scatter()
3. On first import the kernels compile once; all subsequent runs are fast.

Compatibility
-------------
  Python >= 3.10, Numba >= 0.57, NumPy >= 1.24.
  No other runtime dependencies.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# 1.  r_type_scatter  —  replaces RTypeAdaptor.r_type_scatter()
# ---------------------------------------------------------------------------

@njit
def r_type_scatter(S, a_vals, b_vals):
    """
    Compute b = S @ a in-place, avoiding NumPy allocation overhead.

    Parameters
    ----------
    S      : float64 ndarray (n, n)   scattering matrix
    a_vals : float64 ndarray (n,)     incident waves
    b_vals : float64 ndarray (n,)     reflected waves (written in-place)

    Notes
    -----
    For n <= 9 (all PE78 adaptors), the explicit double loop is faster than
    numpy's @ operator inside a per-sample Python callback because it avoids
    temporary allocation and Python dispatch overhead.
    """
    n = S.shape[0]
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += S[i, j] * a_vals[j]
        b_vals[i] = acc


# ---------------------------------------------------------------------------
# 2.  par_scatter — star-junction S matrices (replaces _par_scatter)
# ---------------------------------------------------------------------------

@njit
def par_scatter_n(G, Gt, S):
    """
    Fill S for an n-port star junction (S[i,j] = 2*G[j]/Gt − δ(i,j)).

    Parameters
    ----------
    G  : float64 ndarray (n,)      port conductances 1/Rp
    Gt : float64                   total conductance  sum(G)
    S  : float64 ndarray (n, n)    output (written in-place)
    """
    n = G.shape[0]
    inv_Gt = 2.0 / Gt
    for i in range(n):
        for j in range(n):
            S[i, j] = G[j] * inv_Gt
        S[i, i] -= 1.0


# Convenience wrappers for the fixed sizes used in PE78 (avoids allocation
# of G array at call site — caller passes pre-allocated buffers).

@njit
def par_scatter_4(G, Gt, S):
    """4-port star junction (TR4 collector node in cymb.py)."""
    inv_Gt = 2.0 / Gt
    for i in range(4):
        for j in range(4):
            S[i, j] = G[j] * inv_Gt
        S[i, i] -= 1.0


@njit
def par_scatter_5(G, Gt, S):
    """5-port star junction (ENV node in cymb.py, TR3 node in snare.py)."""
    inv_Gt = 2.0 / Gt
    for i in range(5):
        for j in range(5):
            S[i, j] = G[j] * inv_Gt
        S[i, i] -= 1.0


# ---------------------------------------------------------------------------
# 3.  Newton-Raphson Ebers-Moll BJT solver
# ---------------------------------------------------------------------------

@njit
def nr_ebers_moll(G_fixed, I_fixed, V0, Is_b, Vt, max_iter=20, tol=1e-12):
    """
    Solve the single-port Ebers-Moll KCL equation via Newton-Raphson.

    Equation
    --------
        f(V)  = V * G_fixed + I_fixed + Is_b * (exp(V/Vt) − 1) = 0
        f'(V) = G_fixed + Is_b * exp(V/Vt) / Vt  > 0   (monotone)

    f is strictly monotone, so NR has a unique solution and converges
    quadratically from any starting point.  Typically 3-5 iterations.

    Parameters
    ----------
    G_fixed  : float   sum of fixed-port conductances (excluding BJT port)
    I_fixed  : float   sum of G_k * a_vals[k] for fixed ports
    V0       : float   warm-start guess (previous sample's solution)
    Is_b     : float   Ebers-Moll base saturation current [A]
    Vt       : float   thermal voltage [V] (~26 mV at 300 K)
    max_iter : int     maximum iterations (default 20)
    tol      : float   convergence threshold on |dV| [V] (default 1e-12)

    Returns
    -------
    V : float   solved node voltage [V]
    """
    V = V0
    for _ in range(max_iter):
        arg = V / Vt
        # clamp to avoid overflow in exp
        if arg > 40.0:
            arg = 40.0
        elif arg < -40.0:
            arg = -40.0
        ex  = np.exp(arg)
        Ib  = Is_b * (ex - 1.0)
        f   = V * G_fixed + I_fixed + Ib
        fp  = G_fixed + Is_b * ex / Vt
        dV  = -f / fp
        V  += dV
        if dV * dV < tol * tol:
            break
    return V


# ---------------------------------------------------------------------------
# 4.  omega4 — Wright Omega 4th-order approximation  (wdf.Diode kernel)
# ---------------------------------------------------------------------------

@njit
def omega4(x):
    """
    4th-order approximation of the Wright Omega function W(x).

    Used inside the WDF Diode / DiodePair reflected-wave formula:
        b = a + 2*R*Is − 2*Vt * omega4(log(R*Is/Vt) + a/Vt + R*Is/Vt)

    Piecewise polynomial matching wdf.Diode.omega4() exactly:
        x < x1  →  y = 0
        x1 <= x < x2  →  cubic in x
        x >= x2  →  y = x − log(x)   then one NR refinement step

    Parameters
    ----------
    x : float

    Returns
    -------
    float  approximation of W(x)
    """
    x1 = -3.341459552768620
    x2 =  8.0
    a  = -1.314293149877800e-3
    b  =  4.775931364975583e-2
    c  =  3.631952663804445e-1
    d  =  6.313183464296682e-1

    if x < x1:
        y = 0.0
    elif x < x2:
        y = d + x * (c + x * (b + x * a))
    else:
        y = x - np.log(x)

    return y - (y - np.exp(x - y)) / (y + 1.0)


# ---------------------------------------------------------------------------
# 5.  bilinear_hp — 1st-order bilinear high-pass IIR step
# ---------------------------------------------------------------------------

@njit
def bilinear_hp_step(x, x_prev, y_prev, alpha):
    """
    One step of the 1st-order bilinear high-pass filter.

    Difference equation
    -------------------
        y[n] = alpha * (x[n] - x[n-1]) + (2*alpha - 1) * y[n-1]

    where  alpha = k / (1 + k),  k = 2 * fs * tau.

    This is the bilinear-transform equivalent of H(s) = s*tau / (1 + s*tau).
    Used for C28/C29 (noise-to-base coupling) and as the core of iir_scalar.

    Parameters
    ----------
    x      : float   current input sample
    x_prev : float   previous input sample (state)
    y_prev : float   previous output sample (state)
    alpha  : float   filter coefficient (pre-computed from tau, fs)

    Returns
    -------
    y : float   filtered output
    """
    return alpha * (x - x_prev) + (2.0 * alpha - 1.0) * y_prev


# ---------------------------------------------------------------------------
# 6.  iir_scalar — Direct-form-I scalar 1st-order IIR  (C31 output HP)
# ---------------------------------------------------------------------------

@njit
def iir_scalar(x, x_prev, y_prev, b0, b1, a1):
    """
    Direct-form-I 1st-order IIR filter step.

    Difference equation
    -------------------
        y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]

    where  a1 = _a_c31[1]  (note: sign convention matches cymb.py / snare.py).

    Parameters
    ----------
    x, x_prev : float   current and previous input
    y_prev    : float   previous output
    b0, b1    : float   numerator coefficients (from _make_c31_coeffs)
    a1        : float   denominator coefficient  (note: positive value stored)

    Returns
    -------
    y : float   filtered output
    """
    return b0 * x + b1 * x_prev - a1 * y_prev
