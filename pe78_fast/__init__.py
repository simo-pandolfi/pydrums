#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
pe78_fast — Numba JIT acceleration package for the PE78 drum engine
====================================================================
Importa questo package PRIMA di istanziare qualsiasi strumento PE78.

Ordine di applicazione delle patch:
  1. pywdf_patch  — RTypeAdaptor.r_type_scatter, Diode.omega4
  2. cymb_patch   — PE78_Cymbals_Maracas._solve_env_ebers_moll,
                    _par_scatter (cymb module), C31 IIR
  3. snare_patch  — PE78_Snare._solve_base_ebers_moll,
                    _par_scatter (snare module), C31 IIR
  4. twint_patch  — TwinTDrum._scattering_logic, TwinTDrum.process_sample

Uso minimo:
    import pe78_fast   # applica tutte le patch e compila i kernel JIT

Oppure patch selettive (es. solo rtype + snare):
    from pe78_fast import pywdf_patch, snare_patch
"""

from pe78_fast import pywdf_patch   # noqa: F401  (1)
from pe78_fast import cymb_patch    # noqa: F401  (2)
from pe78_fast import snare_patch   # noqa: F401  (3)
from pe78_fast import twint_patch   # noqa: F401  (4)
