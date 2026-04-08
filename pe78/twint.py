#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE Tonal Drum Voices — Edge Detector + Twin-T Oscillator
=========================================================
Practical Electronics, January 1978 — Fig. 7.
Conga added from M252-M253 SGS-Ates TechnicalNotes 131

Voices: Bass Drum (IC4a), Hi Bongo (IC4b), Low Bongo (IC4c),
        Claves (IC4d), Conga (same structure, different capacitor values).

Structure:
  EdgeDetector: converts a digital gate into a positive trigger spike.
  TwinTDrum:    damped Twin-T oscillator with a CD4011 NAND gate used as
                a linear inverter with DC bias and variable decay (VR4).

Physical topology — 4 nodes, 9 WDF ports:
  n0 = NB_R    : midpoint of R11-R12 (C5 shunt to GND)
  n1 = NC_C    : midpoint of C6-C7  (R16 shunt to GND, trigger injection)
  n2 = NAND_IN : CD4011 pins 1+2
  n3 = NAND_OUT: CD4011 pin 3

  First T  (resistive branch): [NAND_IN]──[R11]──┬──[R12]──[NAND_OUT]
                                                 └──[C5]──[GND]
                                                
  Second T (capacitive branch):[NAND_IN]──[C6]──┬──[C7]──[NAND_OUT]
                                                └──[R16]──[GND]
                                                └──[trigger spike]

  Bias+Decay: [NAND_IN]──[R10 + VR4]──[NAND_OUT]
  
  Output:     [NAND_OUT]──[R41]──[instrument bus]

WDF port order (indices in components list and scattering matrix A):
  0:r11  1:r12  2:c5  3:c6  4:c7  5:r16  6:vtrig  7:r_bias  8:r_out

Inverter model (small-signal):
  V_n3 = a_gain * V_n2,  output impedance r_o
  Typical CD4011 at Vdd=5V: a_gain in [-30, -50], r_o in [500, 2000] Ohm

Scattering matrix correction for the inverter (row n3 of M):
  M[3,3] += 1/r_o
  M[3,2] -= a_gain/r_o   (a_gain < 0 -> increases M[3,2])

Bias resistor semantics:
  r_fb = R10 + VR4  (total value in Ohm, not VR4 alone)
  R10 = 150 kOhm (fixed bias resistor, sets the DC operating point)
  VR4 = potentiometer controlling bias, decay; range 0 ... ~470 kOhm
  Physical range of r_bias: 150 kOhm (VR4=0) ... ~620 kOhm (VR4 max)
  Typical operating point: VR4 ~= 150 kOhm -> r_bias ~= 300 kOhm
"""

import numpy as np
import pywdf.core.wdf as wdf
import pywdf.core.rtype as rtype


# =============================================================================
#  EdgeDetector
# =============================================================================
class EdgeDetector:
    """
    Edge Detector — converts a digital gate to a positive trigger spike.
    (Practical Electronics, January 1978, Fig. 6 pulse shaper section)

    Topology:
      TRIG -> [Node_T1: VS || R14]──[C8]──> [Node_T2: || R15]──[R13]──>[D2]

      R14 (12 kOhm): pull-down, anchors Node_T1 to GND (M253 open-drain output)
      C8  (0.15 uF): AC differentiator;  tau1 = C8*R15 ~= 7 ms
      R15 (47 kOhm): shunt Node_T2 to GND
      R13 (27 kOhm): current limiter;    tau2 = C8*R13 ~= 4 ms
      D2  (1N4148):  half-wave rectifier, passes only positive spikes
                     (rising edge of the trigger gate)

    Parameters
    ----------
    fs : int
        Sample rate in Hz.
    r_vals : list of float
        [R14, R15, R13].
    c_val : float
        C8 capacitance in Farad.
    """

    def __init__(self, fs, r_vals, c_val):
        r14 = float(r_vals[0])
        r15 = float(r_vals[1])
        r13 = float(r_vals[2])

        self.vs  = wdf.ResistiveVoltageSource(1e-3)  # near-ideal voltage source
        self.r14 = wdf.Resistor(r14)
        self.c8  = wdf.Capacitor(float(c_val), fs)
        self.r15 = wdf.Resistor(r15)
        self.r13 = wdf.Resistor(r13)

        # WDF tree: root = Diode (nonlinear element, open port)
        p_t1       = wdf.ParallelAdaptor(self.vs, self.r14)   # Node_T1: VS || R14
        s_c8       = wdf.SeriesAdaptor(p_t1, self.c8)         # series C8
        p_t2       = wdf.ParallelAdaptor(s_c8, self.r15)      # Node_T2: || R15
        self.s_r13 = wdf.SeriesAdaptor(p_t2, self.r13)        # series R13
        self.diode = wdf.Diode(self.s_r13, Is=2.52e-9)        # D2 1N4148

    def process_sample(self, v_trigger: float) -> float:
        """
        Compute one sample of the edge detector output.

        Parameters
        ----------
        v_trigger : float
            Gate input voltage (e.g. 0 V or 4.5 V from M253).

        Returns
        -------
        float
            Voltage across R13 (~= 0 V at rest, positive spike on rising edge).
            Pass directly to TwinTDrum.process_sample().
        """
        self.vs.set_voltage(v_trigger)
        self.diode.accept_incident_wave(self.s_r13.propagate_reflected_wave())
        self.s_r13.accept_incident_wave(self.diode.propagate_reflected_wave())
        return self.r13.wave_to_voltage()


# =============================================================================
#  TwinTDrum
# =============================================================================
class TwinTDrum:
    """
    Damped Twin-T oscillator with CD4011 NAND gate as linear inverter.
    (Practical Electronics 1978, Fig. 7 — Bass Drum and variants)

    Parameters
    ----------
    fs : int
        Sample rate in Hz.
    r_params : list of float
        [R11, R12, R16].
    c_params : list of float
        [C5, C6, C7].
    vr4 : float
        VR4 potentiometer value in Ohm.
        Physical range: 0 ... ~470 kOhm.
    r_load : float
        R41 — output isolation resistor to instrument bus (390 kOhm typical).
    r_o : float, optional
        CD4011 output impedance in Ohm (default 1000).
    a_gain : float, optional
        CD4011 open-loop small-signal gain, must be negative (default -40).
    """

    # Fixed bias resistor R10 (sets inverter DC operating point)
    R10 = 150e3     # [Ohm]

    def __init__(self, fs, r_params, c_params, vr4, r_load,
                 r_o=1000.0, a_gain=-40.0):
        r11 = float(r_params[0])
        r12 = float(r_params[1])
        r16 = float(r_params[2])
        c5  = float(c_params[0])
        c6  = float(c_params[1])
        c7  = float(c_params[2])

        self.r_o    = float(r_o)
        self.a_gain = float(a_gain)
        self._vr4   = float(vr4)

        # WDF components
        self.r11          = wdf.Resistor(r11)
        self.r12          = wdf.Resistor(r12)
        self.c5           = wdf.Capacitor(c5, fs)
        self.c6           = wdf.Capacitor(c6, fs)
        self.c7           = wdf.Capacitor(c7, fs)
        self.r16          = wdf.Resistor(r16)
        self.r_trigger_in = wdf.ResistiveVoltageSource(27e3)            # Z = R13
        self.r_bias       = wdf.Resistor(self.R10 + self._vr4)         # R10 + VR4
        self.r_out        = wdf.Resistor(float(r_load))                 # R41

        self.components = [
            self.r11,           # port 0
            self.r12,           # port 1
            self.c5,            # port 2
            self.c6,            # port 3
            self.c7,            # port 4
            self.r16,           # port 5
            self.r_trigger_in,  # port 6
            self.r_bias,        # port 7
            self.r_out,         # port 8
        ]

        self.drum_osc = rtype.RootRTypeAdaptor(
            self.components, self._scattering_logic
        )

    # -----------------------------------------------------------------------
    # Public setter — control-rate parameter change
    # -----------------------------------------------------------------------
    def set_vr4(self, vr4: float):
        """
        Update VR4 at control rate (maps directly to the physical potentiometer).

        The total bias resistance becomes R10 + VR4. Decay increases with VR4:
          low VR4  -> inverter closer to saturation  -> shorter decay
          high VR4 -> inverter more linear            -> longer decay
        Physical range: 0 Ohm (VR4=0) ... ~470 kOhm.

        The scattering matrix S is recomputed automatically via the pywdf
        impedance propagation mechanism.

        Parameters
        ----------
        vr4 : float
            New VR4 value in Ohm.
        """
        self._vr4 = float(vr4)
        self.r_bias.set_resistance(self.R10 + self._vr4)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------
    def process_sample(self, v_trigger: float) -> float:
        """
        Compute one audio sample.

        Parameters
        ----------
        v_trigger : float
            Output of EdgeDetector.process_sample() — positive spike,
            injected into NC_C (n1) via r_trigger_in (impedance = R13).

        Returns
        -------
        float
            Voltage at NAND_OUT (n3) read through R41 (port 8).
            Note: output starts with a negative half-cycle (inverting gate).
            AC coupling downstream makes polarity perceptually irrelevant.
        """
        self.r_trigger_in.set_voltage(v_trigger)

        for idx, comp in enumerate(self.components):
            self.drum_osc.a_vals[idx] = comp.propagate_reflected_wave()

        self.drum_osc.r_type_scatter()

        for idx, comp in enumerate(self.components):
            comp.accept_incident_wave(self.drum_osc.b_vals[idx])

        return self.r_out.wave_to_voltage()

    # -----------------------------------------------------------------------
    # Internal: scattering matrix with inverter feedback
    # -----------------------------------------------------------------------
    def _scattering_logic(self, adapter):
        """
        WDF scattering matrix for the R-type adaptor.

        S = 2 * diag(G) * A^T * M^-1 * A - I
        M = A * diag(G) * A^T  (passive KCL, corrected for inverter feedback)

        Matrix A (4 nodes x 9 ports):
          rows : n0=NB_R, n1=NC_C, n2=NAND_IN, n3=NAND_OUT
          cols : r11, r12, c5, c6, c7, r16, vtrig, r_bias, r_out
          +1 = positive terminal at node, -1 = negative terminal at node.
          Verification rule: shunt columns sum to +1, series columns sum to 0.

        Inverter feedback correction (row n3 of M):
          M[3,3] += 1/r_o         (current leaving n3 through inverter output)
          M[3,2] -= a_gain/r_o    (amplified current from n2; a_gain < 0)
        """
        R_ports = np.array(adapter.get_port_impedances())
        G       = 1.0 / R_ports

        A = np.array([
        #    r11   r12    c5    c6    c7   r16  vtrig rbias  rout
            [ -1,   1,    1,    0,    0,    0,    0,    0,    0  ],  # n0 NB_R
            [  0,   0,    0,   -1,    1,    1,    1,    0,    0  ],  # n1 NC_C
            [  1,   0,    0,    1,    0,    0,    0,    1,    0  ],  # n2 NAND_IN
            [  0,  -1,    0,    0,   -1,    0,    0,   -1,    1  ],  # n3 NAND_OUT
        ], dtype=float)

        M = A @ np.diag(G) @ A.T           # passive KCL 4x4

        G_inv    = 1.0 / self.r_o
        M[3, 3] += G_inv
        M[3, 2] -= self.a_gain * G_inv     # a_gain < 0 -> increases M[3,2]

        S = 2.0 * np.diag(G) @ A.T @ np.linalg.inv(M) @ A - np.eye(9)
        adapter.set_S_matrix(S)
