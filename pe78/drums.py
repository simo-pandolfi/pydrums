#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE78 Drum Engine — Common interface
=====================================
Wrapper classes providing a uniform tick() interface for all nine voices
of the PE78 / M253 drum machine.

Voices:
  bd     Bass Drum          (AnalogDrum, tonal)
  hb     Hi Bongo           (AnalogDrum, tonal) -- shares trigger bus with SnareDrum
  lb     Low Bongo          (AnalogDrum, tonal)
  cl     Claves             (AnalogDrum, tonal)
  cd     Conga              (AnalogDrum, tonal)
  sd     Snare Drum         (SnareDrum,  noise-based)
  lc     Long Cymbal        (CymbDrum,   noise-based)
  sc     Short Cymbal       (CymbDrum,   noise-based)
  mr     Maracas            (CymbDrum,   noise-based)

Note on hb/sd bus sharing:
  In the M253 sequencer, the Hi Bongo trigger line also drives the Snare Drum.
  Every HB hit activates both voices simultaneously. If the snare is silent,
  the cause is almost certainly the snare model, not the sequencer.

Instrument parameter dictionaries:
  All VR4 values represent the potentiometer alone (not R10+VR4).
  TwinTDrum adds R10=150 kOhm internally.
  Default VR4=150 kOhm -> r_bias = R10+VR4 = 300 kOhm (mid-travel).
"""

import numpy as np

from pe78.snare import PE78_Snare
from pe78.cymb  import PE78_Cymbals_Maracas
from pe78.twint import EdgeDetector, TwinTDrum

NOISE_STD = 0.02    # [V]  default white noise standard deviation


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------
class BaseWDFDrum:
    """Common interface for all PE78 drum voices."""

    def __init__(self, fs):
        self.fs = fs

    def process_sample(self, v_trig, noise_sample=0.0):
        """Advance the model by one sample. Override in subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  Noise-based voices
# ---------------------------------------------------------------------------
class SnareDrum(BaseWDFDrum):
    """Snare Drum — PE78 Fig. 6 top section (filtered white noise)."""

    def __init__(self, fs, noise_std=NOISE_STD):
        super().__init__(fs)
        self.model     = PE78_Snare(fs)
        self.noise_std = noise_std

    def tick(self, v_trig, noise_sample=None):
        """
        Parameters
        ----------
        v_trig : float
            Trigger voltage (4.5 V = on, 0 V = off).
        noise_sample : float or None
            White noise sample (sigma ~= 1). Generated internally if None.
        """
        if noise_sample is None:
            noise_sample = np.random.normal(0, self.noise_std)
        return self.model.process_sample(v_trig, noise_sample)


class CymbDrum(BaseWDFDrum):
    """Long Cymbal + Short Cymbal + Maracas — PE78 Fig. 6 bottom section."""

    def __init__(self, fs, noise_std=NOISE_STD):
        super().__init__(fs)
        self.model     = PE78_Cymbals_Maracas(fs)
        self.noise_std = noise_std

    def tick(self, v_trig_lc, v_trig_sc, v_trig_mr,
                       noise_sample=None):
        """
        Parameters
        ----------
        v_trig_lc, v_trig_sc, v_trig_mr : float
            Trigger voltages for each voice (4.5 V = on, 0 V = off).
        noise_sample : float or None
            White noise sample (sigma ~= 1). Generated internally if None.
        """
        if noise_sample is None:
            noise_sample = np.random.normal(0, self.noise_std)
        return self.model.process_sample(
            v_trig_lc, v_trig_sc, v_trig_mr, noise_sample)

# ---------------------------------------------------------------------------
#  Tonal voices
# ---------------------------------------------------------------------------
class TonalDrum(BaseWDFDrum):
    """
    Tonal drum voice — EdgeDetector + TwinTDrum (damped sinusoidal oscillator).

    Used for: Bass Drum, Hi Bongo, Low Bongo, Claves, Conga.
    """

    def __init__(self, fs, edge_r, edge_c, kick_r, kick_c, kick_vr4, kick_load):
        """
        Parameters
        ----------
        fs : int
            Sample rate in Hz.
        edge_r : list of float
            [R14, R15, R13, R16] for EdgeDetector.
        edge_c : float
            C8 capacitance in Farad for EdgeDetector.
        kick_r : list of float
            [R11, R12, R16] for TwinTDrum.
        kick_c : list of float
            [C5, C6, C7] for TwinTDrum.
        kick_vr4 : float
            VR4 potentiometer value in Ohm (TwinTDrum adds R10 internally).
        kick_load : float
            R41 output isolation resistor in Ohm.
        """
        self.detector = EdgeDetector(fs, edge_r, edge_c)
        self.voice    = TwinTDrum(fs, kick_r, kick_c, kick_vr4, kick_load)

    def tick(self, v_trigger, noise_sample=0.0):
        """
        Parameters
        ----------
        v_trigger : float
            Gate voltage (4.5 V = on, 0 V = off).
        noise_sample : float
            Unused; present for interface uniformity.
        """
        v_spike = self.detector.process_sample(v_trigger)
        return self.voice.process_sample(v_spike)

    def set_decay(self, vr4: float):
        """
        Set VR4 decay potentiometer at control rate.

        Parameters
        ----------
        vr4 : float
            VR4 value in Ohm (0 = shortest decay, ~470 kOhm = longest).
        """
        self.voice.set_vr4(vr4)


# ---------------------------------------------------------------------------
#  Instrument parameter dictionaries
#  kick_vr4 = VR4 alone (TwinTDrum adds R10=150 kOhm internally)
#  kick_vr4 default = 150 kOhm -> r_bias = 300 kOhm (mid-travel)
# ---------------------------------------------------------------------------

BDO_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],  # R14, R15, R13, R16
    'edge_c':    150e-9,                         # C8
    'kick_r':    [68000, 68000, 10000],          # R11, R12, R16
    'kick_c':    [150e-9, 47e-9, 47e-9],         # C5, C6, C7
    'kick_vr4':  150000,                         # VR4 (r_bias = R10+VR4 = 300 kOhm)
    'kick_load': 3900000,                        # R41
}

CONGA_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],
    'edge_c':    56e-9,
    'kick_r':    [68000, 68000, 10000],
    'kick_c':    [56e-9, 18e-9, 18e-9],
    'kick_vr4':  150000,
    'kick_load': 3900000,
}

HBONGO_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],
    'edge_c':    33e-9,
    'kick_r':    [68000, 68000, 10000],
    'kick_c':    [33e-9, 10e-9, 10e-9],
    'kick_vr4':  150000,
    'kick_load': 3900000,
}

LBONGO_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],
    'edge_c':    47e-9,
    'kick_r':    [68000, 68000, 10000],
    'kick_c':    [47e-9, 15e-9, 15e-9],
    'kick_vr4':  150000,
    'kick_load': 3900000,
}

CLAVES_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],
    'edge_c':    4.7e-9,
    'kick_r':    [68000, 68000, 10000],
    'kick_c':    [4.7e-9, 1.5e-9, 1.5e-9],
    'kick_vr4':  150000,
    'kick_load': 3900000,
}