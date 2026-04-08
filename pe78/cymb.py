#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE78 — Long Cymbal + Short Cymbal + Maracas
MNA/Werner Wave Digital Filter implementation
==============================================
Source: Practical Electronics, Jan 1978, Fig. 6 — bottom section.

Architecture overview
---------------------
Sub-circuit A — Envelope node (ENV):
  RootRTypeAdaptor([ser_lc, ser_sc, rvsrc_mr_port, r52, r_pi_port])

  LC branch:
    ser_lc = Series(Parallel(rvsrc_lc, c23), r47)
    Topology: PIN21──[R_src]──┬──[R47, 1MΩ]──> ENV
                              └─[C23, 0.33µF]── GND

  SC branch (rvsrc_sc voltage inverted: source on GND side):
    ser_sc = Series(r44, Parallel(c24, Series(r42, rvsrc_sc)))
    Topology: PIN20──[R_src]──[R42, 10kΩ]──┬──[R44, 1MΩ]──> ENV
                                            └─[C24, 68nF]── GND

  MR → ENV port (one-sample delayed Thevenin of the MR branch):
    rvsrc_mr_port: R_th = R46, V_th = −V_C26[n-1]
    R_th = R46 only (not R45+R46): C26 sits at n_MR and separates
    R45 (D9 side) from R46 (ENV side) in the Thevenin seen from ENV.

  r52           : ENV pull-down to GND (1 MΩ)
  r_pi_port     : TR4 base impedance (updated each sample; see below)

Sub-circuit B — TR4 collector:
  RootRTypeAdaptor([l2, r54, vr9l, tr4_src])
  Driven by V_env from sub-circuit A (one-sample delay).

MR noise path — D9 nonlinearity
---------------------------------
The Maracas branch has two stages separated by D9:

  n_A (anode side, modelled as C25 IIR Thevenin):
    PIN19──[R_src]──┬──[R43, 22kΩ]── GND
                    └──[C25, 0.22µF]── n_A ──[D9→]──[R45, 100kΩ]──> n_MR
    n_A──[←D8]── GND  (D8 clamps n_A to −0.6 V; not modelled — V_C25
                        is clamped to 0 V by the IIR, which is equivalent
                        for V_C25 ≥ 0 as in normal operation.)

  n_MR (hold node):
    n_MR──[R46, 470kΩ]──> ENV
         └──[C26, 0.1µF]── GND

C25 is modelled as a first-order IIR (not a WDF port) to avoid the WDF
freeze that occurs when D9 enters reverse bias.  R43 sets the Thevenin
impedance of the anode source: R_par = R_src ‖ R43.

D9 is a pywdf Diode root connected to:
  anode  ← Thevenin(R_par_mr, −V_c25)  from C25 IIR
  cathode → r45 → Parallel(c26, Series(r46, rvsrc_env_th))

TR4 base loading — Ebers-Moll Newton-Raphson solver (Werner MNA §IV)
----------------------------------------------------------------------
After each root_env.compute(), the reflected-wave vector root.a_vals encodes
the Thevenin state of all fixed ports.  Using Millman's theorem (pywdf sign):

    V_env = −I_fixed / G_total
    where  G_fixed = Σ_{k≠rpi} G_k,   I_fixed = Σ_{k≠rpi} G_k · a_vals_k

The BC547B base current follows the Shockley diode equation:
    Ib(V) = Is_b · (exp(V/Vt) − 1)

Substituting G_total = G_fixed + Ib(V)/V into the Millman equation yields:

    f(V) = V · G_fixed + I_fixed + Is_b · (exp(V/Vt) − 1) = 0

This is solved per-sample with Newton-Raphson:
    f'(V) = G_fixed + Is_b · exp(V/Vt) / Vt  > 0 for all V  (monotone)

Monotonicity of f guarantees:
  - A unique solution from any starting point (no bistability)
  - Quadratic NR convergence (typically 3-5 iterations)
  - No hard Vbe threshold → smooth gm(t) → no amplitude artifact at cutoff

Is_b is calibrated so that the DC operating point with LC trigger matches
LTspice:  V_env_ss ≈ 620 mV.  This gives:
  Is_b = (V_th − V_target) / (R_th · (exp(V_target/Vt) − 1))
       ≈ 7.035 × 10⁻¹⁷ A   (close to IS/BF of the BC547B SPICE model)

After solving, r_pi_eff = hFE·Vt/Ic is stored for the D9-branch ENV Thevenin
and applied to r_pi_port before the next sample's scatter.

Quantitative agreement with LTspice reference
----------------------------------------------
  V_ENV during LC trigger ≈ 620 mV  (LTspice: ~620 mV, error < 1 %)
  No period-2 oscillation at any trigger voltage.
  SC output decay smoother: no hard cutoff at Vbe threshold.
"""

import numpy as np
import pywdf.core.wdf as _wdf
import pywdf.core.rtype as _rtype

Resistor               = _wdf.Resistor
Capacitor              = _wdf.Capacitor
Inductor               = _wdf.Inductor
ResistiveVoltageSource = _wdf.ResistiveVoltageSource
SeriesAdaptor          = _wdf.SeriesAdaptor
ParallelAdaptor        = _wdf.ParallelAdaptor
Diode                  = _wdf.Diode
RootRTypeAdaptor       = _rtype.RootRTypeAdaptor


# ---------------------------------------------------------------------------
# Shared parallel scattering (star junction)
# ---------------------------------------------------------------------------
def _par_scatter(adaptor):
    """S[i][j] = 2·G[j]/Gt − δ(i,j).  Sets adaptor.Rp = 1/Gt."""
    G  = [1.0 / p.Rp for p in adaptor.down_ports]
    Gt = sum(G)
    n  = len(G)
    S  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i][j] = 2.0 * G[j] / Gt - (1.0 if i == j else 0.0)
    adaptor.set_S_matrix(S)
    adaptor.Rp = 1.0 / Gt


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class PE78_Cymbals_Maracas:
    """
    PE78 Long Cymbal / Short Cymbal / Maracas — Wave Digital Filter.

    -----------------------------
    TR4 base loading is resolved each sample via an Ebers-Moll Newton-Raphson
    solver derived from Millman's theorem on the parallel WDF junction.
    The Shockley equation Ib = Is_b·(exp(V/Vt)−1) is monotone → unique root,
    no bistability, no hard Vbe threshold, smooth gm decay.

    V_ENV converges to ~620 mV during LC trigger (LTspice reference: ~620 mV).

    Usage
    -----
        drum = PE78_Cymbals_Maracas(fs)
        out  = drum.process_sample(v_lc, v_sc, v_mr, noise)

    Observable state (updated after each process_sample call):
        _V_env  : ENV node voltage [V]
        _V_c23  : C23 voltage [V]
        _V_c24  : C24 voltage [V]
        _V_c26  : C26 voltage [V]
        _V_col  : TR4 collector voltage [V]
    """

    # -----------------------------------------------------------------------
    # Physical parameters  (PE 1978 Fig. 6, bottom section)
    # -----------------------------------------------------------------------
    V_ON  = 4.5        # M253 output when conducting [V]
    R_ON  = 350.0      # M253 on-impedance [Ω]
    R_OFF = 10e6       # M253 tri-state impedance [Ω]

    R47   = 1e6        # LC: C23 discharge resistor; tau = R47·C23 = 330 ms
    C23   = 0.33e-6

    R42   = 10e3       # SC: series current limiter
    C24   = 68e-9      # SC: hold cap; tau = R44·C24 = 68 ms
    #C24   = 270e-9    # to increase tau time to about 170 ms

    R44   = 1e6

    R43   = 22e3       # MR: D9 anode pull-down
    C25   = 0.22e-6    # MR: decoupling cap (IIR, not WDF)
    R45   = 100e3      # MR: D9 cathode series
    C26   = 0.1e-6     # MR: hold cap; tau = R46·C26 = 47 ms
    R46   = 470e3

    R52   = 1e6        # ENV pull-down / TR4 base bias

    # TR4 (BC547B)
    Vcc       = 12.0
    R54       = 4700.0
    L2_val    = 100e-3
    C29       = 4.7e-9
    hFE       = 200
    Vt        = 26e-3
    # Ebers-Moll base saturation current, calibrated so that
    # V_ENV_ss ≈ 620 mV with LC trigger (matches LTspice BC547B operating point).
    # Derivation: Is_b = (V_th − V_target) / (R_th · (exp(V_target/Vt) − 1))
    # with V_th ≈ 1200 mV, R_th ≈ 363 kΩ, V_target = 620 mV.
    Is_b      = 7.035e-17  # [A]  ≈ IS/BF of BC547B SPICE model
    # Vbe_th is kept for gm reconstruction in the test script and for the
    # D9-branch ENV Thevenin (used only as a fallback guard rail).
    Vbe_th    = 0.58       # [V]  approximate; not used in the NR solver
    R_SRC_TR4 = R54
    NOISE_AMP = 0.04

    D9_Is = 2.52e-9    # 1N4148

    VR9L_default = 195e3
    R55   = 22e3
    C31   = 0.1e-6
    Rload = 10e3

    # r_pi guard rails
    R_PI_MAX = 10e6    # open-base / cut-off
    R_PI_MIN = 500.0   # deep-saturation floor

    # r_pi port index inside root_env.down_ports  (set in _build_env_node)
    _RPI_PORT_IDX = 4

    # -----------------------------------------------------------------------
    def __init__(self, fs: int, vr9l: float = None):
        self.fs  = int(fs)
        self._dt = 1.0 / fs
        self.VR9L = float(vr9l) if vr9l is not None else self.VR9L_default

        self._on_lc = self._on_sc = self._on_mr = False

        # C25 IIR state (D9 anode)
        self._V_c25  = 0.0

        # One-sample delays
        self._V_env  = 0.0    # V_ENV[n-1]  (used to drive TR4 collector)
        self._V_c26  = 0.0    # C26[n-1]    (MR → ENV Thevenin voltage)

        # r_pi_eff from last NR solve — feeds D9-branch ENV Thevenin
        self._r_pi_applied = 10e6   # starts at open-base (cut-off)

        # C29 HP state (TR4 base noise)
        self._v_base   = 0.0
        self._v_base_x = 0.0
        
        # C31 output HP
        self._c31_x = 0.0
        self._c31_y = 0.0
        self._b_c31, self._a_c31 = self._make_c31_coeffs()

        # Observable
        self._V_c23  = 0.0
        self._V_c24  = 0.0
        self._V_col  = 0.0

        self._build_env_node()
        self._build_mr_tree()
        self._build_tr4_node()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def set_vr9l(self, value: float):
        """Update VR9L at control rate (between triggers, not per-sample)."""
        self.VR9L = float(value)
        self._vr9l.set_resistance(self.VR9L)
        self._b_c31, self._a_c31 = self._make_c31_coeffs()

    def process_sample(self, v_trig_lc: float, v_trig_sc: float,
                       v_trig_mr: float, noise_sample: float = 0.0) -> float:
        """
        Advance the circuit by one sample.

        Parameters
        ----------
        v_trig_lc, v_trig_sc, v_trig_mr : float
            M253 trigger voltages: 0.0 V = idle, 4.5 V = active.
        noise_sample : float
            White noise sample (σ ≈ 1), scaled internally by NOISE_AMP.

        Returns
        -------
        float
            Audio output [V] after C31 high-pass coupling.
        """
        on_lc = v_trig_lc > 0.5
        on_sc = v_trig_sc > 0.5
        on_mr = v_trig_mr > 0.5

        # ---- 1. Source impedance transitions (only on edges) ------------
        if on_lc != self._on_lc:
            self._rvsrc_lc.set_resistance(self.R_ON if on_lc else self.R_OFF)
            self._on_lc = on_lc
        if on_sc != self._on_sc:
            self._rvsrc_sc.set_resistance(self.R_ON if on_sc else self.R_OFF)
            self._on_sc = on_sc
        if on_mr != self._on_mr:
            self._rvsrc_mr.set_resistance(self.R_ON if on_mr else self.R_OFF)
            self._on_mr = on_mr

        # ---- 2. Source voltages ----------------------------------------
        self._rvsrc_lc.set_voltage(self.V_ON if on_lc else 0.0)
        self._rvsrc_sc.set_voltage(-(self.V_ON if on_sc else 0.0))
        self._rvsrc_mr.set_voltage(self.V_ON if on_mr else 0.0)

        # ---- 3. C25 IIR (D9 anode; decoupled to prevent WDF freeze) ----
        R_src_mr     = self._rvsrc_mr.Rval
        V_src_mr     = self.V_ON if on_mr else 0.0
        R_par_mr     = R_src_mr * self.R43 / (R_src_mr + self.R43)
        V_anode_tgt  = V_src_mr * self.R43 / (R_src_mr + self.R43)
        tau_c25      = self.C25 * R_par_mr
        self._V_c25 += self._dt / (tau_c25 + self._dt) * (V_anode_tgt - self._V_c25)

        self._rvsrc_anode_th.set_resistance(R_par_mr)
        self._rvsrc_anode_th.set_voltage(-self._V_c25)   # −sign: WDF port convention

        # ---- 4. MR → ENV Thevenin (one-sample delayed C26) -------------
        # R_th = R46: ENV sees n_MR through R46 only.  R45 is on the D9 side
        # of n_MR and is separated from ENV by C26, so it does not appear in
        # the Thevenin.  V_th = −V_C26 (sign: R52 pulls ENV to GND).
        self._rvsrc_mr_port.set_resistance(self.R46)
        self._rvsrc_mr_port.set_voltage(-self._V_c26)

        # ---- 5. D9 cathode-side Thevenin (ENV seen from D9 cathode) ----
        #  Includes r_pi from the PREVIOUS scatter for correct ENV impedance.
        R_src_lc = self._rvsrc_lc.Rval
        R_src_sc = self._rvsrc_sc.Rval
        R_env_th = 1.0 / (
            1.0 / (self.R47 + R_src_lc)
            + 1.0 / (self.R42 + self.R44 + R_src_sc)
            + 1.0 / self.R52
            + 1.0 / self._r_pi_applied       # TR4 base loading
        )
        self._rvsrc_env_th.set_resistance(R_env_th)
        self._rvsrc_env_th.set_voltage(self._V_env)

        # ---- 6. Step D9 / MR branch ------------------------------------
        self._d9.accept_incident_wave(self._tree_mr.propagate_reflected_wave())
        self._tree_mr.accept_incident_wave(self._d9.propagate_reflected_wave())
        self._V_c26 = -self._c26.wave_to_voltage()

        # ---- 7. Step ENV node (parallel WDF junction) ------------------
        self._root_env.compute()

        # ---- 8. Ebers-Moll NR: resolve V_env, gm, r_pi -------------------
        V_env_new, gm_new, r_pi_new = self._solve_env_ebers_moll()

        self._V_env  = V_env_new
        self._V_c23  = self._c23.wave_to_voltage()
        self._V_c24  = self._c24.wave_to_voltage()

        # Update r_pi_port for next sample's scatter
        if abs(r_pi_new - self._r_pi_applied) / self._r_pi_applied > 1e-4:
            self._r_pi_port.set_resistance(r_pi_new)
            self._r_pi_applied = r_pi_new

        # ---- 9. TR4 collector (sub-circuit B, driven by V_env_new) -----
        V_col       = self._step_tr4(V_env_new, gm_new, noise_sample * self.NOISE_AMP)
        self._V_col = V_col

        # ---- 10. C31 IIR high-pass output coupling ----------------------
        x = V_col
        y = (self._b_c31[0] * x
             + self._b_c31[1] * self._c31_x
             - self._a_c31[1] * self._c31_y)
        self._c31_x = x
        self._c31_y = y
        return y

    # -----------------------------------------------------------------------
    # Private: TR4 loading — Ebers-Moll Newton-Raphson solver
    # -----------------------------------------------------------------------
    def _solve_env_ebers_moll(self):
        """
        Resolve V_env and the TR4 operating point via Ebers-Moll NR.

        Equation to solve each sample (Werner 2015 §IV, single-port NL):
            f(V) = V · G_fixed + I_fixed + Is_b · (exp(V/Vt) − 1) = 0
            f'(V) = G_fixed + Is_b · exp(V/Vt) / Vt          > 0  always

        G_fixed, I_fixed come from the current root_env.a_vals (leaf reflected
        waves), which are already advanced by root_env.compute().

        Monotonicity of f' guarantees a unique root and global convergence of
        Newton-Raphson from any starting point.  In practice 3-5 iterations
        suffice for convergence to 1 pV.

        Returns
        -------
        V_env  : float   ENV node voltage [V]
        gm     : float   TR4 transconductance hFE·Is_b·exp(V/Vt)/Vt  [A/V]
        r_pi   : float   small-signal base resistance hFE·Vt/Ic  [Ω]
        """
        a_vals  = self._root_env.a_vals
        Rps     = [p.Rp for p in self._root_env.down_ports]
        idx_rpi = self._RPI_PORT_IDX

        G_fixed = 0.0
        I_fixed = 0.0
        for k in range(len(Rps)):
            if k == idx_rpi:
                continue
            gk = 1.0 / Rps[k]
            G_fixed += gk
            I_fixed += gk * a_vals[k]

        # Newton-Raphson: start from previous V_env for fast convergence
        V = self._V_env
        Vt = self.Vt
        Is_b = self.Is_b
        for _ in range(20):
            ex  = np.exp(np.clip(V / Vt, -40.0, 40.0))
            Ib  = Is_b * (ex - 1.0)
            f   = V * G_fixed + I_fixed + Ib
            fp  = G_fixed + Is_b * ex / Vt
            dV  = -f / fp
            V  += dV
            if abs(dV) < 1e-12:
                break

        # Collector current and small-signal parameters
        ex_sol = np.exp(np.clip(V / Vt, -40.0, 40.0))
        Ic     = self.hFE * Is_b * ex_sol
        Ic     = min(Ic, (self.Vcc - 0.2) / self.R54)
        Ic     = max(Ic, 1e-15)
        gm     = Ic / Vt
        r_pi   = float(np.clip(self.hFE * Vt / Ic, 500.0, 10e6))

        return V, gm, r_pi

    # -----------------------------------------------------------------------
    # Private: TR4 collector step (sub-circuit B)
    # -----------------------------------------------------------------------
    def _step_tr4(self, V_env: float, gm: float, noise_b1: float) -> float:
        """
        Advance TR4 collector one sample.  Returns V_col [V].

        Parameters
        ----------
        V_env : float   ENV node voltage, used to compute r_base for C29 filter.
        gm    : float   Transconductance from Ebers-Moll NR solve [A/V].
        noise_b1 : float  Scaled noise sample entering TR4 base via C29.
        """
        # r_base = r_pi || R52, with r_pi = hFE·Vt/Ic = hFE/gm
        if gm > 1e-12:
            r_pi   = self.hFE / gm
            r_base = r_pi * self.R52 / (r_pi + self.R52)
        else:
            r_base = self.R52

        # C29 — AC coupling high-pass, first-order bilinear IIR.
        # Analog prototype: H(s) = sτ / (1 + sτ),  τ = C29·r_base
        # Bilinear discretisation (α = 2fs·τ/(1+2fs·τ), pole p = 2α−1):
        #   H(z) = α·(1 − z⁻¹) / (1 − (2α−1)·z⁻¹)
        # Direct form I:  y[n] = α·(x[n]−x[n-1]) + (2α−1)·y[n-1]
        # fc = 1/(2π·τ) ≈ 3.4 kHz at the nominal operating point
        #   (r_base ≈ 9.8 kΩ, Ic ≈ 527 µA, V_ENV ≈ 633 mV).
        tau_c29      = self.C29 * r_base
        k            = 2.0 * self.fs * tau_c29       # coefficiente bilineare
        alpha_hp     = k / (1.0 + k)                 # polo del HP
        x_new        = noise_b1 - self._v_base_x     # differenza ingresso
        self._v_base = alpha_hp * x_new + (2.0 * alpha_hp - 1.0) * self._v_base
        self._v_base_x = noise_b1                     # stato ritardato

        self._tr4_src.set_voltage(gm * self._v_base * self.R_SRC_TR4)
        self._tr4_node.compute()
        return self._r54.wave_to_voltage()

    # -----------------------------------------------------------------------
    # Private: ENV node construction
    # -----------------------------------------------------------------------
    def _build_env_node(self):
        """
        ENV = RootRTypeAdaptor([ser_lc, ser_sc, rvsrc_mr_port, r52, r_pi_port])
        Index of r_pi_port = _RPI_PORT_IDX = 4.
        """
        # LC branch
        self._rvsrc_lc = ResistiveVoltageSource(self.R_OFF)
        self._c23      = Capacitor(self.C23, self.fs)
        self._r47      = Resistor(self.R47)
        ser_lc = SeriesAdaptor(
            ParallelAdaptor(self._rvsrc_lc, self._c23),
            self._r47)

        # SC branch (source on GND side → inverted voltage)
        self._rvsrc_sc = ResistiveVoltageSource(self.R_OFF)
        self._r42      = Resistor(self.R42)
        self._c24      = Capacitor(self.C24, self.fs)
        self._r44      = Resistor(self.R44)
        ser_sc = SeriesAdaptor(
            self._r44,
            ParallelAdaptor(self._c24, SeriesAdaptor(self._r42, self._rvsrc_sc)))

        # MR → ENV Thevenin (one-sample delayed C26)
        # Thevenin seen by ENV looking into the MR branch:
        #   R_th = R46  (R46 connects n_MR to ENV; R45 is on the D9 side of C26)
        # When D9 conducts the path is R_src→R43→C25→D9→R45→n_MR→R46→ENV,
        # but C26 sits at n_MR and separates R45 from R46 in the Thevenin
        # seen from ENV.  R45 therefore does not appear in R_th.
        self._rvsrc_mr_port = ResistiveVoltageSource(self.R46)
        self._rvsrc_mr_port.set_voltage(0.0)

        # ENV pull-down
        self._r52 = Resistor(self.R52)

        # TR4 base port — r_pi_eff from Ebers-Moll NR (updated each sample)
        self._r_pi_port = Resistor(10e6)   # starts at open-base

        # Verify index assumption
        ports = [ser_lc, ser_sc, self._rvsrc_mr_port, self._r52, self._r_pi_port]
        assert ports[self._RPI_PORT_IDX] is self._r_pi_port, \
            f"_RPI_PORT_IDX={self._RPI_PORT_IDX} mismatch — update the class constant"

        self._root_env = RootRTypeAdaptor(ports, _par_scatter)

    # -----------------------------------------------------------------------
    # Private: MR branch (D9 as root)
    # -----------------------------------------------------------------------
    def _build_mr_tree(self):
        """
        D9 (1N4148) root — topology follows Fig. 6 schematic:

          n_A  (anode side, modelled as C25 IIR Thevenin):
            PIN19 ──[R_src]──┬──[R43, 22kΩ]── GND
                             └──[C25, 0.22µF]── n_A ──[D9 →]──[R45, 100kΩ]──> n_MR
            n_A ──[← D8]── GND   (D8 not modelled: V_C25 is clamped to 0 V
                                   by the IIR, which is equivalent for V_C25 ≥ 0.
                                   D8 would clamp to −0.6 V but that region is
                                   never reached in normal operation.)

          n_MR (hold node = junction R45 / C26 / R46):
            n_MR ──[R46, 470kΩ]──> ENV
                 └──[C26, 0.1µF]── GND

        anode  ← Thevenin(R_par_mr, −V_c25)  from C25 IIR
        cathode → [R45] → n_MR → Parallel(c26, Series(r46, rvsrc_env_th))

        rvsrc_env_th models the Thevenin of ENV seen from n_MR:
          V_th  = V_env[n-1]
          R_th  = R46 + (R52 ‖ R_lc ‖ R_sc ‖ r_pi)
        (R46 is in series between n_MR and the ENV node.)
        """
        # _rvsrc_mr is not wired into any WDF node; it exists solely to
        # track the current R_ON/R_OFF value of the MR trigger source so
        # that process_sample() can compute R_par_mr = R_src ‖ R43.
        self._rvsrc_mr = ResistiveVoltageSource(self.R_OFF)
        self._r45        = Resistor(self.R45)
        self._c26        = Capacitor(self.C26, self.fs)
        self._r46        = Resistor(self.R46)

        self._rvsrc_anode_th = ResistiveVoltageSource(self.R43)
        self._rvsrc_anode_th.set_voltage(0.0)

        R_env_th_init = 1.0 / (
            1.0 / self.R47
            + 1.0 / (self.R42 + self.R44)
            + 1.0 / self.R52
        )
        self._rvsrc_env_th = ResistiveVoltageSource(R_env_th_init)
        self._rvsrc_env_th.set_voltage(0.0)

        r46_env      = SeriesAdaptor(self._r46, self._rvsrc_env_th)
        cathode_tree = SeriesAdaptor(
            self._r45,
            ParallelAdaptor(self._c26, r46_env))
        self._tree_mr = SeriesAdaptor(self._rvsrc_anode_th, cathode_tree)
        self._d9      = Diode(self._tree_mr, Is=self.D9_Is, Vt=25.85e-3)

    # -----------------------------------------------------------------------
    # Private: TR4 collector node (sub-circuit B)
    # -----------------------------------------------------------------------
    def _build_tr4_node(self):
        """
        TR4 collector = RootRTypeAdaptor([l2, r54, vr9l, tr4_src])
        All ports shunt to AC-GND (+12 V bypassed).
        V_col is read from r54.wave_to_voltage().
        """
        self._l2      = Inductor(self.L2_val, self.fs)
        self._r54     = Resistor(self.R54)
        self._vr9l    = Resistor(self.VR9L)
        self._tr4_src = ResistiveVoltageSource(self.R_SRC_TR4)
        self._tr4_src.set_voltage(0.0)

        self._tr4_node = RootRTypeAdaptor(
            [self._l2, self._r54, self._vr9l, self._tr4_src],
            _par_scatter)

    # -----------------------------------------------------------------------
    # Private: C31 IIR coefficients
    # -----------------------------------------------------------------------
    def _make_c31_coeffs(self):
        """
        Bilinear IIR coefficients for the output high-pass filter.

        Physical path: V_col → [VR9L + R55] → C31 → Rload → GND
        Transfer function:
            H(s) = s·τ_z / (1 + s·τ_p)
            τ_z = C31·Rload          (zero)
            τ_p = C31·(VR9L+R55+Rload)  (pole)

        Returns (b, a) as length-2 arrays for the direct-form I recursion:
            y[n] = b[0]·x[n] + b[1]·x[n-1] − a[1]·y[n-1]
        """
        Rs    = self.VR9L + self.R55
        Rl    = self.Rload
        C     = self.C31
        tau_p = C * (Rs + Rl)
        tau_z = C * Rl
        k     = 2.0 * self.fs
        b0    =  k * tau_z
        b1    = -k * tau_z
        a0    = 1.0 + k * tau_p
        a1    = 1.0 - k * tau_p
        return np.array([b0 / a0, b1 / a0]), np.array([1.0, a1 / a0])