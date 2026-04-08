# py78drums

A Python implementation of the **DIY drum machine circuit** from 
*Nuova Elettronica* n. 60–61 June–July 1978, modelled using Wave Digital Filters (WDF).

This is a **research project in progress**, developed as a study in preparation
for a Rust port as an audio plugin. The models are validated against LTspice
simulations; they have not been verified against a physical PE78 unit.

---

## Motivation

This project has a personal origin. In the early 1980s I built a DIY electronic
drum machine following the circuit published in *Nuova Elettronica* n. 60–61
(June–July 1978). The unit was eventually lost, and decades later I chose it as
a study subject for virtual analog synthesis with Wave Digital Filters — partly
for the technical interest, partly for the personal connection.

When researching the circuit online I first found the *Practical Electronics*
version (January 1978), which I remembered as very similar, and started the
implementation from that schematic. I later found the original *Nuova Elettronica*
circuit, which differs in one key aspect: it uses the **SGS-Ates M252AA** rhythm
generator instead of the M253 used in PE78, and includes one additional instrument
voice. I also found SGS-Ates Technical Notes 131, which appears to be the common
source from which most DIY drum circuits of that era descend — several near-identical
designs have been found in other publications of the period.

The implementation therefore combines the PE78 circuit (used as the primary
schematic reference, component name and LTspice validation target) with M252AA inspired rhythm patterns,
which correspond to the sequencer in the original *Nuova Elettronica* build.

---

## Background

The PE78 circuit produces nine percussion voices using two distinct synthesis
approaches:

- **Twin-T oscillators** (Bass Drum, Hi Bongo, Low Bongo, Conga Drum, Claves, Conga) —
  damped sinusoidal oscillators triggered by edge-detection circuits (Fig. 7)
- **Filtered white noise** (Snare Drum, Long Cymbal, Short Cymbal, Maracas) —
  transistor noise generators with RC envelope networks amplified by a BJT
  common-emitter stage (Fig. 6)

The WDF methodology models each subcircuit at the component level, preserving
the physical behaviour of reactive elements (capacitors, inductors) and
nonlinear elements (diodes, BJT transconductance).

The sequencer (wdf_rithm.py) adopts the 32-step, two-bar architecture 
inspired by the SGS-Ates M252AA chip, allowing for rhythmic variations 
between the first and second bar, while featuring a custom set of original patterns.

---

## Project structure

```
pydrums/
├── ltspice/           # Circuit simulation
├── pe78/
│   ├── cymb.py        # Long Cymbal / Short Cymbal / Maracas (Fig. 6 bottom)
│   ├── snare.py       # Snare Drum (Fig. 6 top)
│   ├── twint.py       # Twin-T oscillator voices (BD, HB, LB, CL, CD)
│   └── drums.py       # Uniform tick() interface for all nine voices
├── sequencer/
│   └── wdf_rithm.py   # rhythm sequencer — renders patterns to WAV
│   └── patterns.py    # Original rhythms inspired by the 70s and 80s.
├── bench/             # Level verification and audio output tools
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
numba
scipy
soundfile
git+https://github.com/gusanthon/pywdf
```

To pre-compile the Numba JIT functions at install time (avoids first-run
latency):

```bash
python -c "import pe78.cymb; import pe78.snare; import pe78.twint"
```

---

## Usage

### Render a rhythm pattern

`wdf_rithm.py` is now interactive. You can run it and follow the on-screen menu to choose a rhythm, tempo, and duration:

```bash
python wdf_rithm.py
```

You can also pass parameters directly as command-line arguments:
```
python wdf_rithm.py <rhythm_number> [bpm] [bars]
```

Example:

```
# Generate 8 bars of the 3rd rhythm at 128 BPM
python wdf_rithm.py 3 128 8
```


When you run a synthesis, the script performs the following steps:

*Snare Diagnosis*: It automatically runs a 500ms test on the Snare model to check peak voltage and envelope conduction.  

*WDF Model Initialization*: It initializes the specific Virtual Analog trees for the instruments required by the pattern (Bass Drum, Snare, Bongos, Congas, etc.).  

*Sample-by-Sample Processing*: The engine calculates every single sample at the defined Sample Rate (default 48kHz), emulating the physical behavior of the original circuit.  

*Output*: A normalized 16-bit WAV file is saved in the project root with a descriptive name (e.g., `pe78_rock_1_120bpm_4bar.wav`).

> **Technical Note on Triggering**: In the original PE78 circuit, the Hi Bongo (HB) bus is physically hardwired to the Snare Drum (SD) trigger. This means that every snare hit also triggers the Hi Bongo voice, a detail reproduced in this implementation.

### Example Audio
If you want to listen to the models without running the code, pre-generated simulation results (4-bar loops) are available in the sequencer/ folder. These files demonstrate the current state of the Virtual Analog synthesis for all rhythmic patterns.

### Use individual voices

```python
from pe78.drums import CymbDrum, SnareDrum, TonalDrum, BDO_PARAMS

FS = 48000

# Cymbal section (Long Cymbal, Short Cymbal, Maracas share one instance)
cymb = CymbDrum(FS)
sample = cymb.tick(v_trig_lc=4.5, v_trig_sc=0.0, v_trig_mr=0.0)

# Snare
snare = SnareDrum(FS)
sample = snare.tick(v_trig=4.5)

# Bass Drum (tonal Twin-T voice)
bd = TonalDrum(FS, **BDO_PARAMS)
sample = bd.tick(v_trigger=4.5)
```

Trigger voltage follows the M252/M253 output characteristics: `4.5 V` when
active, **tri-state (high impedance)** when inactive — the output does not
pull to 0 V but floats. In the model this is represented as `R_OFF = 10 MΩ`
on the source impedance. The `tick()` call advances the model by one sample
and returns the audio output in volts.

---

## Validation status

| Voice | Validated against LTspice | Notes |
|---|---|---|
| Bass Drum | ✓ frequency, decay | |
| Hi / Low Bongo | ✓ frequency, decay | |
| Claves | ✓ | Numerical sensitivity at 48 kHz — see Known Limitations |
| Conga | ✓ frequency, decay | |
| Snare | ✓ envelope timing ≤1%, spectrum ±3 dB | Phase 16 |
| Long Cymbal | ✓ envelope timing | Level ~0.9× LTspice |
| Short Cymbal | ⚠ envelope too short (~32 ms vs 181 ms) | See Known Limitations |
| Maracas | ✓ envelope timing | Level ~2× LTspice |

---

## Known limitations

- **Short Cymbal envelope duration** is ~32 ms in the model vs ~181 ms in
  LTspice. The discrepancy is due to the static Ebers-Moll TR4 model vs the
  dynamic Gummel-Poon model used by LTspice. Increasing `C24` to ~260 nF
  compensates empirically.
- **TR4 model** uses static Ebers-Moll (instantaneous response). Dynamic
  internal capacitances (Cbc, Cbe) are not modelled and would extend cymbal
  envelope durations toward LTspice reference values.
- **VR9U / VR9L** are independent parameters. The physical VR9 is a single
  220 kΩ pot with the constraint VR9U + VR9L ≈ 220 kΩ, which is not enforced
  in the model.
- **Claves** at 48 kHz: WDF port impedances reach MΩ range; 96 kHz reduces
  bilinear transform artefacts.
- **Output levels** are calibrated against the LTspice simulation, not a
  physical PE78 unit.
- **NOISE_AMP = 0.04** is matched to the LTspice behavioural noise source,
  not to a physical measurement of TR2 (BC108B in reverse breakdown).

---

## References

- *Practical Electronics*, January 1978, pp. 24–25 — PE78 circuit schematic
- *Nuova Elettronica*, n. 60–61, June–July 1978 — original build reference
- SGS-Ates Technical Notes 131 — M252/M253 sequencer, common source of
  most DIY drum circuits of the era
- SGS-Ates - MOS and special CMOS/MOS 1st edition issued Nov.1979 - M252/M253 and other IC rythms table
- Fettweis, A. (1986). Wave Digital Filters: Theory and Practice. *IEEE*
- Werner, K. J., Nangia, V., Smith, J. O., & Abel, J. S. (2015). Resolving
  Wave Digital Filters with Multiple/Multiport Nonlinearities. *IEEE WASPAA*
- Werner, K. J., Bernardini, A., Smith, J. O., & Sarti, A. (2018). Modeling
  Circuits with Arbitrary Topologies and Active Linear Multiports Using Wave
  Digital Filters. *IEEE Transactions on Circuits and Systems I*, 65(12),
  4233–4246. https://doi.org/10.1109/TCSI.2018.2837912

---

## Dependencies

- [pywdf](https://github.com/gusanthon/pywdf) — WDF framework (MIT)
  Anthon, G., Lizarraga-Seijas, X., Font, F. (2023). *PYWDF: an open source
  library for prototyping and simulating wave filter circuits in Python*. DAFx23.
- [chowdsp_wdf](https://github.com/Chowdhury-DSP/chowdsp_wdf) — C++ WDF library
  on which pywdf is based (Jatin Chowdhury)
- numpy, numba, scipy, soundfile

---

## AI assistance

This project was developed with extensive use of **Claude (Anthropic)** as an
AI assistant, contributing to WDF circuit decomposition, scattering matrix
derivation, debugging of sign conventions and impedance mismatches, level
calibration, and validation methodology.

---

## License

Copyright © 2026 Simone Pandolfi.

This project is dual-licensed under the [MIT License](LICENSE-MIT) and the [Apache License, Version 2.0](LICENSE-APACHE). 
You may choose which license you'd like to use.