#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
wdf_rithm.py — PE 1978 Noise Drums: sequencer ritmico M253 AC
==============================================================
Sequencer a 16/32 step con i ritmi del M253 AC e tutti gli strumenti
fisici del circuito PE 1978.

Strumenti:
  bd     Bass Drum          (TonalDrum)
  hb     Hi Bongo           (TonalDrum) — condivide il bus trigger con SD
  lb     Low Bongo          (TonalDrum)
  sd     Snare Drum         (SnareDrum) — stesso trigger di HB
  cl     Claves             (TonalDrum)
  cd     Conga              (TonalDrum)
  lc     Long Cymbal        (CymbDrum)
  sc     Short Cymbal       (CymbDrum)
  mr     Maracas            (CymbDrum)

Note sul cablaggio M253:
  Il bus SD e il bus HB sono collegati in parallelo: ogni trigger HB
  attiva ENTRAMBI gli strumenti. Il pattern 'hb' attiva quindi
  TonalDrum(hb) + SnareDrum(sd) contemporaneamente.

  LC, SC e MR condividono un unico nodo ENV (CymbDrum): vengono
  processati con un'unica chiamata tick() per campione, esattamente
  come nel circuito fisico.
"""

import sys
import os
import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Path setup — garantisce che il progetto sia trovato
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Patch Numba — un solo import, applica tutto in ordine corretto
# ---------------------------------------------------------------------------
import pe78_fast   # noqa: F401  compila i kernel JIT e patcha pywdf + modelli

# ---------------------------------------------------------------------------
# Import strumenti (DOPO le patch)
# ---------------------------------------------------------------------------
from pe78.drums import (
    TonalDrum, SnareDrum, CymbDrum,
    BDO_PARAMS, CONGA_PARAMS, HBONGO_PARAMS, LBONGO_PARAMS, CLAVES_PARAMS,
)

from patterns_M252AA import PATTERNS_M252AA


# ---------------------------------------------------------------------------
# Parametri di sessione
# ---------------------------------------------------------------------------
FS          = 48_000
BPM_DEFAULT = 100
STEPS       = 32
HIT_SAMPLES = 960       # durata impulso trigger ≈ 20 ms @ 48 kHz

# Gain per strumento (bilanciamento relativo sul master bus)
GAIN = {
    'bd':   1.00,
    'hb':   1.00,
    'lb':   1.00,
    'cd':   1.00,
    'cl':   1.00,
    'sd':   4.00,    # SnareDrum ha uscita tipicamente più bassa
    'cymb': 4.00,    # CymbDrum idem
}


# ---------------------------------------------------------------------------
# Diagnostica snare
# ---------------------------------------------------------------------------
def _diagnose_snare(fs):
    """
    Istanzia uno SnareDrum isolato, invia un singolo colpo e
    stampa peak e durata attiva per verificare il modello.
    """
    dr  = SnareDrum(fs)
    n   = int(0.5 * fs)
    out = np.zeros(n)
    for i in range(n):
        out[i] = dr.tick(4.5 if i < HIT_SAMPLES else 0.0)
    peak   = np.max(np.abs(out))
    active = int(np.sum(np.abs(out) > peak * 0.01)) if peak > 1e-9 else 0
    print(f"  Snare diagnostic: peak={peak:.5f} V, "
          f"campioni attivi={active} ({active / fs * 1000:.0f} ms)")
    if peak < 1e-4:
        print("  → SEGNALE QUASI NULLO: D7 probabilmente non conduce. "
              "Lo snare sarà silenzioso.")
    else:
        print("  → Snare OK")


# ---------------------------------------------------------------------------
# Sintesi
# ---------------------------------------------------------------------------
def synthesize(rhythm_name, bpm=BPM_DEFAULT, num_bars=4,
               latin_mode=True, fs=FS):
    """
    Sintetizza il ritmo campione per campione, come un motore audio real-time.

    Parameters
    ----------
    rhythm_name : str   chiave in PATTERNS_M252AA
    bpm         : int   battiti per minuto
    num_bars    : int   numero di battute da sintetizzare
    latin_mode  : bool  True=Claves su OUT3, False=Hi Bongo aggiuntivo
    fs          : int   sample rate in Hz

    Returns
    -------
    np.ndarray  segnale audio normalizzato a ±0.5
    """
    if rhythm_name not in PATTERNS_M252AA:
        raise ValueError(
            f"Ritmo '{rhythm_name}' non trovato.\n"
            f"Disponibili: {list(PATTERNS_M252AA.keys())}"
        )

    pattern = PATTERNS_M252AA[rhythm_name]

    # step_samples = durata di una semicroma in campioni
    step_samples  = int(60.0 / bpm / 4 * fs)
    pattern_steps = pattern.get('steps', STEPS)
    total_samples = step_samples * pattern_steps * num_bars
    master        = np.zeros(total_samples, dtype=np.float64)

    print(f"\nDiagnosi snare:")
    _diagnose_snare(fs)

    # -----------------------------------------------------------------------
    # 1. Estrazione sequenze dal pattern
    #    Valori assenti → sequenza di zeri lunga pattern_steps
    # -----------------------------------------------------------------------
    def _seq(key):
        return pattern.get(key, [0] * pattern_steps)

    seq_bd = _seq('bd')
    seq_hb = _seq('hb')
    seq_sd = _seq('sd')   # se assente nel pattern, tutti 0
    seq_lb = _seq('lb')
    seq_cl = _seq('cl')
    seq_cd = _seq('cd')
    seq_lc = _seq('lc')
    seq_sc = _seq('sc')
    seq_mr = _seq('mr')

    # Il trigger dello snare segue il bus HB (wiring M253).
    # Se il pattern dichiara anche 'sd' esplicitamente, OR logico:
    seq_sd_trig = [max(h, s) for h, s in zip(seq_hb, seq_sd)]

    # -----------------------------------------------------------------------
    # 2. Rilevamento strumenti usati
    # -----------------------------------------------------------------------
    has_bd   = any(seq_bd)
    has_hb   = any(seq_hb)
    has_lb   = any(seq_lb)
    has_cl   = any(seq_cl)
    has_cd   = any(seq_cd)
    has_sd   = any(seq_sd_trig)    # SnareDrum attivo se c'è HB o SD
    has_cymb = any(seq_lc) or any(seq_sc) or any(seq_mr)

    print(f"\nStrumenti attivi: "
          + ", ".join(k for k, v in {
              'bd': has_bd, 'hb': has_hb, 'lb': has_lb,
              'cl': has_cl, 'cd': has_cd, 'sd': has_sd,
              'cymb': has_cymb,
          }.items() if v))

    # -----------------------------------------------------------------------
    # 3. Inizializzazione WDF trees
    # -----------------------------------------------------------------------
    print(f"Inizializzazione strumenti per '{rhythm_name}'...")

    bd_m  = TonalDrum(fs, **BDO_PARAMS)    if has_bd   else None
    hb_m  = TonalDrum(fs, **HBONGO_PARAMS) if has_hb or has_sd else None
    lb_m  = TonalDrum(fs, **LBONGO_PARAMS) if has_lb   else None
    cl_m  = TonalDrum(fs, **CLAVES_PARAMS) if has_cl   else None
    cd_m  = TonalDrum(fs, **CONGA_PARAMS)  if has_cd   else None
    sd_m  = SnareDrum(fs)                  if has_sd   else None
    cymb_m = CymbDrum(fs)                  if has_cymb else None

    rng = np.random.default_rng(42)   # seed fisso per riproducibilità

    # -----------------------------------------------------------------------
    # 4. Main audio loop — campione per campione
    # -----------------------------------------------------------------------
    print(f"Sintesi tick-by-tick: {num_bars} battute @ {bpm} BPM "
          f"({total_samples} campioni)...")

    for i in range(total_samples):
        step_idx    = (i // step_samples) % pattern_steps
        pos_in_step = i % step_samples
        in_trig     = pos_in_step < HIT_SAMPLES
        trig_v      = 4.5 if in_trig else 0.0

        # Campione di rumore condiviso (bus TR2 — snare + cymbal)
        noise = rng.standard_normal()

        mix = 0.0

        # Bass Drum
        if bd_m is not None:
            v = trig_v if seq_bd[step_idx] else 0.0
            mix += bd_m.tick(v) * GAIN['bd']

        # Hi Bongo (TonalDrum) — triggerato dal bus HB
        if hb_m is not None:
            v = trig_v if seq_hb[step_idx] else 0.0
            mix += hb_m.tick(v) * GAIN['hb']

        # Snare Drum — triggerato dal bus HB (cablaggio M253)
        if sd_m is not None:
            v = trig_v if seq_sd_trig[step_idx] else 0.0
            mix += sd_m.tick(v, noise_sample=noise) * GAIN['sd']

        # Low Bongo
        if lb_m is not None:
            v = trig_v if seq_lb[step_idx] else 0.0
            mix += lb_m.tick(v) * GAIN['lb']

        # Claves
        if cl_m is not None:
            v = trig_v if seq_cl[step_idx] else 0.0
            mix += cl_m.tick(v) * GAIN['cl']

        # Conga
        if cd_m is not None:
            v = trig_v if seq_cd[step_idx] else 0.0
            mix += cd_m.tick(v) * GAIN['cd']

        # Cymbal + Maracas (circuito condiviso, unica chiamata)
        if cymb_m is not None:
            v_lc = trig_v if seq_lc[step_idx] else 0.0
            v_sc = trig_v if seq_sc[step_idx] else 0.0
            v_mr = trig_v if seq_mr[step_idx] else 0.0
            mix += cymb_m.tick(v_lc, v_sc, v_mr,
                               noise_sample=noise) * GAIN['cymb']

        master[i] = mix

        # Avanzamento a schermo ogni secondo circa
        if i % fs == 0:
            bar = i // (step_samples * pattern_steps) + 1
            print(f"  battuta {bar}/{num_bars}...", end='\r')

    print()   # newline dopo i \r

    # -----------------------------------------------------------------------
    # 5. Normalizzazione finale (−6 dBFS → 0.5)
    # -----------------------------------------------------------------------
    peak = np.max(np.abs(master))
    if peak > 1e-9:
        master = master / peak * 0.5
    else:
        print("[AVVISO] Master bus silenzioso — controlla gli strumenti.")

    print(f"Sintesi completata. Peak pre-norm: {peak:.5f} V")
    return master


# ---------------------------------------------------------------------------
# Salvataggio WAV
# ---------------------------------------------------------------------------
def save_wav(data, path, fs=FS):
    out = (data * 32767).astype(np.int16)
    wavfile.write(path, fs, out)
    print(f"Salvato: {path}")


# ---------------------------------------------------------------------------
# Main interattivo
# ---------------------------------------------------------------------------
def main():
    rhythm_list = list(PATTERNS_M252AA.keys())

    print("\n" + "=" * 54)
    print("  PE 1978 Noise Drums — Sequencer M253 AC")
    print("=" * 54)
    for i, name in enumerate(rhythm_list, 1):
        print(f"  {i:2d}.  {name}")
    print("   0.  Esci")
    print("=" * 54)
    print("  Uso: <numero> [bpm] [battute]   es: 3 128 8")
    print()

    if len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:])
    else:
        raw = input("Scelta: ").strip()

    parts = raw.split()
    if not parts or parts[0] == '0':
        print("Uscita.")
        return

    try:
        idx = int(parts[0])
        if not (1 <= idx <= len(rhythm_list)):
            raise ValueError
    except ValueError:
        print(f"Scelta non valida: '{parts[0]}'")
        return

    rhythm_name = rhythm_list[idx - 1]
    bpm         = int(parts[1]) if len(parts) > 1 else BPM_DEFAULT
    num_bars    = int(parts[2]) if len(parts) > 2 else 4

    print(f"\nRitmo: {rhythm_name}  |  {bpm} BPM  |  {num_bars} battute")

    audio = synthesize(rhythm_name, bpm=bpm, num_bars=num_bars)

    slug  = rhythm_name.lower().replace(' ', '_')
    fname = f"pe78_{slug}_{bpm}bpm_{num_bars}bar.wav"
    save_wav(audio, fname)


if __name__ == "__main__":
    main()
