#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE78 — Snare Drum
Wave Digital Filter implementation
=====================================
Source: Practical Electronics, Jan 1978, Fig. 6 — snare section.

Architecture overview
---------------------
The circuit is partitioned into three sub-circuits, following
the same methodology used in cymb.py (PE78_Cymbals_Maracas).

Sub-circuit A — Envelope  (IIR scalar, not WDF)
  TR5 (BC108B) emitter follower buffers the SNARE_DRUM trigger.
  D7  (1N4148) charges C27 when V_emitter > V_C27 (ideal diode, no 0.7V
  drop, consistent with the D9 treatment in cymb.py).

    tau_on  = R49 * C27                    ≈ 0.110 ms  (fast attack)
    tau_off = (R50 + R51) * C27            = 100   ms  (slow decay)

  tau_off is computed with TR3 off, which gives the maximum decay time.
  When TR3 is conducting r_pi loads R50+R51 and the effective tau is
  shorter (~51 ms at full drive).  This approximation matches the one
  used in cymb.py for C23/C24 and is accepted as a known limitation.

  V_env = V_C27 * R51 / (R50 + R51)   [V]   (R50 = R51 = 1 MOhm)

Sub-circuit B — TR3 base node  (Ebers-Moll Newton-Raphson)
  The base node is a Thevenin junction of:
    · R50  from node_ENV  (envelope source)
    · R51  to GND         (pull-down / bias)
    · TR3 Vbe junction    (Ebers-Moll nonlinearity)
  C28 noise coupling is handled separately as a bilinear HP IIR.

  With G_fixed = 1/R50 + 1/R51  and  I_fixed = -V_env/R50
  (negative: envelope pushes current INTO the node), the KCL equation is:

      f(V)  = V * G_fixed + I_fixed + Is_b * (exp(V/Vt) - 1) = 0
      f'(V) = G_fixed + Is_b * exp(V/Vt) / Vt  > 0  (monotone)

  Monotonicity guarantees a unique root and quadratic NR convergence.
  Typically 3-5 iterations suffice for convergence to < 1 pV.
  Structurally identical to _solve_env_ebers_moll() in cymb.py.

Sub-circuit C — TR3 collector  (WDF RootRTypeAdaptor)
  Similar to _build_tr4_node() in cymb.py but with an additional port
  for C30 (0.02 uF, collector to GND), which is absent in the cymbal circuit:

      RootRTypeAdaptor([l1, r53, c30, vr9u, tr3_src], _par_scatter)

  where:
    l1      = Inductor(L1_val, fs)          100 mH, ideal (no Rser)
    r53     = Resistor(R53)                 4.7 kOhm  (L1 || R53 to +12V)
    vr9u    = Resistor(VR9U)                25 kOhm  (snare side of VR9 pot)
    tr3_src = ResistiveVoltageSource(R_SRC) Norton->Thevenin: V_th=gm*v_b*R53,
              R_th=R53  (r_o >> R53, approximated as ideal current source)

  Note on Rser: inductors (Coilcraft PCH-45X-107) have Rser=61.7 Ohm.
  Analysis shows this causes < 0.5 dB difference at the L-C30 resonance
  (3559 Hz) and < 0.1 dB elsewhere in the audio band.  Q ~ 2 (heavily
  damped): timbric impact is imperceptible.  Not modelled, consistent
  with cymb.py.

  Note on VR9U: VR9 (220 kOhm) is a balance pot mixing snare and cymbal
  outputs.  With circuits simulated independently, VR9U (snare side,
  110 kOhm) is the AC load seen by the TR3 collector, exactly as VR9L
  does for TR4 in cymb.py.

Output chain
  V_col -> VR9U -> R55 -> C31 -> out
  C31 bilinear IIR HP, identical in form to cymb.py _make_c31_coeffs().

Noise source  (TR2 / B1)
  TR2 (BC108B avalanche) noise is shared between snare and cymbal.
  The caller may pass the same noise_sample to both process_sample() calls.
  If noise_sample is omitted, an internal N(0,1) sample is generated.
  Scaling: NOISE_AMP = 0.04 V (same as cymb.py).

Ebers-Moll calibration
  Is_b is calibrated so that V_base_TR3 at steady state with a 4.5 V
  trigger matches LTspice (BC547B Gummel-Poon model).  Derivation:

      Is_b = (V_th - V_target) / (R_th * (exp(V_target/Vt) - 1))

  with R_th = R50 || R51 = 500 kOhm, V_target = LTspice V_base_TR3_ss.
  Starting value: Is_b = 7.035e-17 A (cymb.py value, BC547B IS/BF).
  Recalibrate against LTspice once the reference waveform is available.

Non-linearity separability
  NL1 (TR5): upstream, no instantaneous coupling to NL2/NL3.
  NL2 (D7):  couples to TR3 only through C27 state (one-sample delay),
             algebraically separable.
  NL3 (TR3): no instantaneous feedback to NL1/NL2.
  Sequential (partitioned) treatment is causally correct.
  Werner (2015) multi-NL and Werner (2018) MNA are not required.

Known approximations (consistent with cymb.py)
  D7 forward voltage drop ignored (ideal diode).
  TR5 modelled as static emitter follower V_E = V_B - Vbe.
  tau_off fixed at 100 ms (TR3-off approximation; actual ~51 ms active).
  Inductor Rser = 61.7 Ohm not modelled.

Component values  (PE 1978 Fig. 6, verified on LTspice ASC)
  Trigger : C22=0.1uF, R48=22kOhm, V_trigger=4.5V (M253 output)
  TR5     : BC108B emitter follower, +4.7V supply
  Envelope: R49=2.2kOhm, C27=0.05uF, R50=1MOhm, R51=1MOhm
  TR3     : BC108B, hFE=200, Vt=26mV
  Noise   : C28=4.7nF (B1 -> base), B1 = 12V + white*0.04V
  Collector: L1=100mH || R53=4.7kOhm, C30=0.02uF (to GND)
  Output  : VR9U=25kOhm, R55=22kOhm, C31=0.1uF
"""

import numpy as np
import pywdf.core.wdf   as _wdf
import pywdf.core.rtype as _rtype

Resistor               = _wdf.Resistor
Capacitor              = _wdf.Capacitor
Inductor               = _wdf.Inductor
ResistiveVoltageSource = _wdf.ResistiveVoltageSource
RootRTypeAdaptor       = _rtype.RootRTypeAdaptor


# ---------------------------------------------------------------------------
# Shared parallel scattering  (identical to cymb.py _par_scatter)
# ---------------------------------------------------------------------------
def _par_scatter(adaptor):
    """S[i,j] = 2*G[j]/Gt - delta(i,j).  Sets adaptor.Rp = 1/Gt."""
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
# Core model
# ---------------------------------------------------------------------------
class PE78_Snare:
    """
    PE78 Snare Drum — Wave Digital Filter.

    TR3 base is resolved each sample via an Ebers-Moll Newton-Raphson
    solver on the base Thevenin node, structurally identical to the TR4
    NR solver in PE78_Cymbals_Maracas (cymb.py).

    Usage
    -----
        snare = PE78_Snare(fs)
        out   = snare.process_sample(v_trigger, noise_sample)

    Observable state (updated after each process_sample call):
        _V_c27  : envelope capacitor voltage [V]
        _V_base : TR3 base voltage (NR solution) [V]
        _V_col  : TR3 collector voltage [V]
    """

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------

    # TR5 emitter follower
    Vbe5      = 0.65        # [V]   static emitter drop

    # Envelope
    R49       = 2_200.0     # [Ohm] D7 -> node_ENV series resistor
    C27       = 0.05e-6     # [F]   hold capacitor
    R50       = 1e6         # [Ohm] node_ENV -> base TR3
    R51       = 1e6         # [Ohm] base TR3 -> GND  (pull-down)

    # TR3 (BC108B)
    Vcc       = 12.0        # [V]   supply
    L1_val    = 100e-3      # [H]   collector inductor  (|| R53 to +12V)
    R53       = 4_700.0     # [Ohm] collector resistor  (|| L1 to +12V)
    C30       = 0.02e-6     # [F]   collector resonance cap (node_C -> GND)
    #                               f_res = 1/(2pi*sqrt(L1*C30)) = 3559 Hz
    #                               Q ~ R53/sqrt(L1/C30) ~ 2.1  (heavily damped)
    C28       = 4.7e-9      # [F]   noise coupling B1 -> base
    hFE       = 200         # [-]   DC current gain
    Vt        = 26e-3       # [V]   thermal voltage at 300 K
    # Ebers-Moll base saturation current.
    # Starting value = cymb.py Is_b (BC547B IS/BF ~ 7e-17 A).
    # Recalibrate: Is_b = (V_th - V_tgt) / (R_th * (exp(V_tgt/Vt) - 1))
    # with R_th = R50||R51 = 500 kOhm, V_tgt = LTspice V_base_TR3_ss.
    Is_b      = 7.035e-17   # [A]

    # WDF source impedance for TR3 Thevenin  (mirrors cymb.py R_SRC_TR4)
    R_SRC_TR3 = R53         # [Ohm]

    # Noise
    NOISE_AMP = 0.04        # [V]   same as cymb.py (shared TR2 bus)

    # Output
    VR9U_default = 25e3    # [Ohm] snare side of VR9 balance pot (calibrated value)
    R55       = 22e3        # [Ohm] series resistor before C31
    C31       = 0.1e-6      # [F]   AC output coupling cap
    Rload     = 10e3        # [Ohm] pre-amp input impedance

    # r_pi guard rails  (mirrors cymb.py)
    R_PI_MAX  = 10e6        # [Ohm] open-base / cut-off
    R_PI_MIN  = 500.0       # [Ohm] deep-saturation floor

    # ------------------------------------------------------------------
    def __init__(self, fs: int, vr9u: float = None, rload: float = None):
        """
        Parameters
        ----------
        fs    : int    Sample rate [Hz].
        vr9u  : float  VR9U wiper resistance [Ohm]; default 25 kOhm (calibrated value).
        rload : float  Pre-amp input impedance [Ohm]; default 10 kOhm.
        """
        self.fs  = int(fs)
        self._dt = 1.0 / self.fs

        if vr9u  is not None:
            self.VR9U_default = float(vr9u)
        if rload is not None:
            self.Rload = float(rload)

        self.VR9U = float(self.VR9U_default)

        # Envelope IIR time constants
        self._tau_on  = self.R49 * self.C27              # ~ 0.110 ms
        self._tau_off = (self.R50 + self.R51) * self.C27 # = 100 ms

        # Envelope state
        self._V_c27 = 0.0

        # NR warm-start state
        self._V_base = 0.0

        # C28 bilinear HP state  (mirrors cymb.py _v_base / _v_base_x)
        self._v_base_hp = 0.0
        self._v_base_x  = 0.0

        # C31 output HP state
        self._c31_x = 0.0
        self._c31_y = 0.0
        self._b_c31, self._a_c31 = self._make_c31_coeffs()

        # Observable
        self._V_col = 0.0

        # Build WDF collector node
        self._build_tr3_node()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_vr9u(self, value: float) -> None:
        """
        Update VR9U wiper resistance at control rate (not per-sample).
        Mirrors cymb.py set_vr9l(): updates WDF port and C31 coefficients.
        """
        self.VR9U = float(value)
        self._vr9u.set_resistance(self.VR9U)
        self._b_c31, self._a_c31 = self._make_c31_coeffs()

    def process_sample(self,
                       v_trigger: float,
                       noise_sample: float = None) -> float:
        """
        Advance the snare circuit by one sample.

        Parameters
        ----------
        v_trigger    : float
            SNARE_DRUM trigger [V].  Typical: 0.0 V idle, 4.5 V active.
        noise_sample : float or None
            White noise sample (sigma ~ 1), scaled internally by NOISE_AMP.
            Pass the same value as the cymbal call to share the TR2 bus.
            If None, a fresh N(0,1) sample is drawn internally.

        Returns
        -------
        float
            Audio output [V] after VR9U -> R55 -> C31 high-pass.
        """
        if noise_sample is None:
            noise_sample = np.random.standard_normal()

        # 1. Envelope: TR5 follower + D7 + C27 IIR
        V_env = self._step_envelope(v_trigger)

        # 2. TR3 base: Ebers-Moll NR
        V_base, gm, r_pi = self._solve_base_ebers_moll(V_env)
        self._V_base = V_base

        # 3. C28 bilinear HP: scale and filter noise
        v_b = self._step_c28(noise_sample * self.NOISE_AMP, gm)

        # 4. TR3 collector: WDF node
        V_col = self._step_tr3_collector(gm, v_b)
        self._V_col = V_col

        # 5. C31 bilinear IIR HP output coupling
        x = V_col
        y = (self._b_c31[0] * x
             + self._b_c31[1] * self._c31_x
             - self._a_c31[1] * self._c31_y)
        self._c31_x = x
        self._c31_y = y
        return y

    # ------------------------------------------------------------------
    # Private: sub-circuit A — envelope
    # ------------------------------------------------------------------

    def _step_envelope(self, v_trigger: float) -> float:
        """
        Advance C27 by one sample.

        TR5 is a static emitter follower: V_E = max(0, V_trigger - Vbe5).
        D7 is ideal (no forward drop, consistent with cymb.py):
          V_E > V_C27  ->  charges   C27 toward V_E  via tau_on
          V_E <= V_C27 ->  discharges C27 toward 0   via tau_off

        Returns  V_env = V_C27 * R51 / (R50 + R51)  [V].
        """
        V_emitter = max(0.0, v_trigger - self.Vbe5)

        if V_emitter > self._V_c27:
            tau, v_target = self._tau_on,  V_emitter
        else:
            tau, v_target = self._tau_off, 0.0

        alpha        = self._dt / (tau + self._dt)   # backward-Euler
        self._V_c27 += alpha * (v_target - self._V_c27)

        return self._V_c27 * (self.R51 / (self.R50 + self.R51))

    # ------------------------------------------------------------------
    # Private: sub-circuit B — TR3 base Ebers-Moll NR
    # ------------------------------------------------------------------

    def _solve_base_ebers_moll(self, V_env: float):
        """
        Resolve TR3 base voltage and operating point via Ebers-Moll NR.

        Base node KCL (currents leaving node positive):
            f(V)  = V * G_fixed + I_fixed + Is_b * (exp(V/Vt) - 1) = 0
            f'(V) = G_fixed + Is_b * exp(V/Vt) / Vt  > 0

        G_fixed = 1/R50 + 1/R51
        I_fixed = -V_env / R50   (envelope injects current into node)

        Structurally identical to cymb.py _solve_env_ebers_moll(), with
        the two-resistor Thevenin (R50, R51) replacing the multi-branch
        ENV junction.

        Returns  (V_base, gm, r_pi).
        """
        G_fixed = 1.0 / self.R50 + 1.0 / self.R51
        I_fixed = -V_env / self.R50     # envelope pushes current in

        Vt   = self.Vt
        Is_b = self.Is_b

        # NR with warm start from previous solution
        V = self._V_base
        for _ in range(20):
            ex = np.exp(np.clip(V / Vt, -40.0, 40.0))
            f  = V * G_fixed + I_fixed + Is_b * (ex - 1.0)
            fp = G_fixed + Is_b * ex / Vt
            dV = -f / fp
            V += dV
            if abs(dV) < 1e-12:
                break

        # Collector current and small-signal parameters
        ex_sol = np.exp(np.clip(V / Vt, -40.0, 40.0))
        Ic     = self.hFE * Is_b * ex_sol
        Ic     = min(Ic, (self.Vcc - 0.2) / self.R53)  # saturation clamp
        Ic     = max(Ic, 1e-15)                         # numerical floor

        gm    = Ic / Vt
        r_pi  = float(np.clip(self.hFE * Vt / Ic,
                               self.R_PI_MIN, self.R_PI_MAX))
        return V, gm, r_pi

    # ------------------------------------------------------------------
    # Private: C28 bilinear high-pass  (noise -> base coupling)
    # ------------------------------------------------------------------

    def _step_c28(self, noise_b1: float, gm: float) -> float:
        """
        Advance C28 bilinear HP by one sample.

        Structurally identical to the C29 HP in cymb.py _step_tr4():
          H(s) = s*tau / (1 + s*tau),  tau = C28 * r_base
          r_base = r_pi || R51

        Bilinear transform (k = 2*fs*tau):
          alpha = k / (1 + k)
          y[n]  = alpha*(x[n] - x[n-1]) + (2*alpha - 1)*y[n-1]

        Returns filtered noise voltage at TR3 base [V].
        """
        if gm > 1e-12:
            r_pi   = self.hFE / gm
            r_base = r_pi * self.R51 / (r_pi + self.R51)
        else:
            r_base = self.R51

        tau_c28 = self.C28 * r_base
        k       = 2.0 * self.fs * tau_c28
        alpha   = k / (1.0 + k)

        x_new           = noise_b1 - self._v_base_x
        self._v_base_hp = alpha * x_new + (2.0 * alpha - 1.0) * self._v_base_hp
        self._v_base_x  = noise_b1
        return self._v_base_hp

    # ------------------------------------------------------------------
    # Private: sub-circuit C — TR3 collector WDF
    # ------------------------------------------------------------------

    def _build_tr3_node(self) -> None:
        """
        Build TR3 collector WDF tree.

        Unlike cymb.py _build_tr4_node(), the snare collector has C30
        (0.02 uF) connected from node_C to GND — absent in the cymbal
        circuit.  C30 creates a resonance with L1 at f_res = 3559 Hz
        (Q ~ 2.1) that shapes the characteristic "crack" of the snare.

        Structure:
            RootRTypeAdaptor([l1, r53, c30, vr9u, tr3_src], _par_scatter)

        All ports shunt to AC-GND (+12V rail bypassed for small signal).
        V_col is read from self._r53.wave_to_voltage().
        """
        self._l1      = Inductor(self.L1_val, self.fs)
        self._r53     = Resistor(self.R53)
        self._c30     = Capacitor(self.C30, self.fs)
        self._vr9u    = Resistor(self.VR9U)
        self._tr3_src = ResistiveVoltageSource(self.R_SRC_TR3)
        self._tr3_src.set_voltage(0.0)

        self._tr3_node = RootRTypeAdaptor(
            [self._l1, self._r53, self._c30, self._vr9u, self._tr3_src],
            _par_scatter)

    def _step_tr3_collector(self, gm: float, v_b: float) -> float:
        """
        Advance TR3 collector WDF node by one sample.

        Norton -> Thevenin conversion:
          V_th = gm * v_b * R_SRC_TR3
          R_th = R_SRC_TR3  (= R53)

        R_SRC_TR3 = R53 << r_o (> 100 kOhm) so the approximation is
        accurate, identical to cymb.py R_SRC_TR4 convention.

        Returns TR3 collector voltage V_col [V].
        """
        self._tr3_src.set_voltage(gm * v_b * self.R_SRC_TR3)
        self._tr3_node.compute()
        return self._r53.wave_to_voltage()

    # ------------------------------------------------------------------
    # Private: C31 bilinear IIR HP coefficients
    # ------------------------------------------------------------------

    def _make_c31_coeffs(self):
        """
        Bilinear IIR HP for C31 output coupling.

        Identical to cymb.py _make_c31_coeffs(), with VR9U in place of VR9L:
          tau_p = C31 * (VR9U + R55 + Rload)
          tau_z = C31 * Rload

        Bilinear transform  (k = 2*fs):
          b = [k*tau_z / (1 + k*tau_p),  -k*tau_z / (1 + k*tau_p)]
          a = [1,                         (1 - k*tau_p) / (1 + k*tau_p)]
        """
        Rs    = self.VR9U + self.R55
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


# ---------------------------------------------------------------------------
# SnareDrum wrapper  —  public interface for wdf_snare_test.py
# ---------------------------------------------------------------------------

class SnareDrum:
    """
    Thin public wrapper around PE78_Snare.

    Exposes the tick() interface expected by wdf_snare_test.py and the
    parent drum machine.

    Shared-noise usage (TR2 bus with cymbal):
    ::
        noise = np.random.standard_normal()
        cymbal_out = cymbal.process_sample(v_lc, v_sc, v_mr, noise)
        snare_out  = snare.tick(v_trigger, noise_sample=noise)
    """

    def __init__(self, fs: int, vr9u: float = None, rload: float = None):
        self.model = PE78_Snare(fs, vr9u=vr9u, rload=rload)

    def set_vr9u(self, value: float) -> None:
        self.model.set_vr9u(value)

    def tick(self,
             v_trigger: float,
             noise_sample: float = None) -> float:
        """
        Per-sample entry point.

        Parameters
        ----------
        v_trigger    : float   Gate voltage [V]: 4.5 V on, 0 V off.
        noise_sample : float   sigma~1 noise (shared TR2 sample); auto if None.

        Returns
        -------
        float   Audio output [V].
        """
        return self.model.process_sample(v_trigger, noise_sample)