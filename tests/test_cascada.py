"""Bateria de regresion de La Cascada de Asimov.

Ejecucion sin dependencias externas (sin pytest):

    python tests/test_cascada.py

Cada test documenta un fallo concreto que ya ocurria y que no debe volver.
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import numpy as np

from motor_generativo import (
    _a_midi,
    _afinidad_tonal,
    _escala_desde_secuencia,
    _nota_a_freq,
    _notas_desde_evento,
    construir_cadena_markov,
    generar_nueva_obra,
)

FALLOS = []
TOTAL = 0


def test(fn):
    global TOTAL
    TOTAL += 1
    try:
        fn()
        print(f"  [OK]   {fn.__name__}")
    except Exception as exc:  # noqa: BLE001
        FALLOS.append((fn.__name__, exc))
        print(f"  [FALLA] {fn.__name__}: {exc}")
    return fn


# --------------------------------------------------------------------------
# Bloque 1: parser canonico (acordes MIDI que se descartaban en silencio)
# --------------------------------------------------------------------------

@test
def test_acorde_cruzo_y_normalizado_coinciden():
    """Bug A: ('60','64','67') resolvia a [] y descartaba el evento entero."""
    crudo = _notas_desde_evento("Acorde_60.64.67")
    normalizado = _notas_desde_evento(("60", "64", "67"))
    assert crudo == [60, 64, 67], crudo
    assert normalizado == [60, 64, 67], normalizado


@test
def test_acorde_produce_frecuencias_multiples():
    """Un acorde debe sonar polifonico, no como una nota monofonica."""
    freq = _nota_a_freq("Acorde_60.64.67")
    assert isinstance(freq, list), type(freq)
    assert len(freq) == 3, freq
    assert all(20 < f < 5000 for f in freq), freq


@test
def test_nota_simple_sigue_siendo_monofonica():
    assert _nota_a_freq("C4") == 261.6255653005986
    assert _nota_a_freq(60) == 261.6255653005986


@test
def test_todas_las_formas_normalizan_igual():
    formas = ["C4", ("C4",), ("60",), 60, 60.0, np.int64(60)]
    resultados = [_a_midi(f) for f in formas]
    assert all(r == (60,) for r in resultados), resultados


@test
def test_descansos_normalizan_a_vacio():
    for silencio in ("REST", "rest", "SILENCIO", "", None, ()):
        assert _a_midi(silencio) == (), silencio


@test
def test_notas_ilegales_se_descartan_sin_romper():
    for basura in ("X999", None, "zzz", (), (None, ""), "Acorde_"):
        _a_midi(basura)


@test
def test_escala_tonal_se_extrae_de_acordes():
    """Bug B: sin escala no hay afinidad, y el Softmax pierde el ancla armonica."""
    escala = _escala_desde_secuencia([
        ("Acorde_60.64.67", 1.0, 1.0),
        ("Acorde_55.59.62", 1.0, 1.0),
        ("C4", 1.0, 1.0),
    ])
    assert escala == {0, 2, 4, 7, 11}, escala


@test
def test_normalizacion_markov_produce_enteros():
    """Los estados de Markov deben ser homogeneos: tupla de ints o 'REST'."""
    modelo = construir_cadena_markov([
        ("Acorde_60.64.67", 1.0, 1.0),
        ("C4", 0.5, 1.0),
        ("REST", 0.25, 0.0),
        ("Acorde_55.59.62", 1.0, 0.5),
    ], orden=2)

    for nota, duracion, velocidad in modelo["estados"]:
        assert nota == "REST" or (isinstance(nota, tuple) and all(isinstance(n, int) for n in nota)), nota
        assert isinstance(duracion, float)
        assert 0.0 <= velocidad <= 1.0


@test
def test_todo_evento_del_acorde_llega_al_sintetizador():
    """Regresion del 57/65: ningun acorde debe perderse antes de sintetizar."""
    secuencia = [
        ("Acorde_60.64.67", 1.0, 1.0),
        ("C4", 1.0, 1.0),
        ("Acorde_55.59.62", 1.0, 1.0),
        ("G4", 1.0, 1.0),
    ]
    modelo = construir_cadena_markov(secuencia, orden=1)
    for nota, _dur, _vel in modelo["estados"]:
        assert _nota_a_freq(nota) is not None, nota


@test
def test_afinidad_tonal_reacciona_a_la_escala():
    en_escala = _afinidad_tonal(("60", "64", "67"), {0, 4, 7})
    fuera_de_escala = _afinidad_tonal(("61", "62", "63"), {0, 4, 7})
    parcial = _afinidad_tonal(("60", "64", "61"), {0, 4, 7})
    assert en_escala == 1.0, en_escala
    assert fuera_de_escala == 0.0, fuera_de_escala
    assert en_escala > parcial > fuera_de_escala, (en_escala, parcial, fuera_de_escala)


@test
def test_afinidades_se_cargan_una_vez_y_se_reutilizan():
    """_notas_desde_evento se llama por candidato y por paso; debe ser barata."""
    import time

    modelo = construir_cadena_markov(
        [("Acorde_60.64.67", 1.0, 1.0), ("C4", 1.0, 1.0), ("E4", 1.0, 1.0)] * 40, orden=2
    )
    inicio = time.time()
    generar_nueva_obra(modelo, 200, 1.8, orden=2)
    transcurrido = time.time() - inicio
    assert transcurrido < 5.0, f"generacion demasiado lenta: {transcurrido:.2f}s"


@test
def test_cascada_completa_no_revienta():
    """Humo: 6 fases con acordes deben terminar y producir audio."""
    import tempfile
    import soundfile as sf

    from motor_generativo import sintetizar_audio_directo

    memoria = {
        "piano": {"instrumento": None,
                  "secuencia": [("Acorde_60.64.67", 0.5, 0.8), ("C4", 0.5, 0.6),
                                ("Acorde_55.59.62", 0.5, 0.7), ("G4", 0.5, 0.5)] * 8},
        "bass": {"instrumento": None,
                 "secuencia": [(36, 1.0, 1.0), (41, 1.0, 0.9), (43, 1.0, 0.8)] * 8},
    }

    with tempfile.TemporaryDirectory() as tmp:
        salida = os.path.join(tmp, "cascade.wav")
        for _fase in range(6):
            nueva = {}
            for pista, datos in memoria.items():
                modelo = construir_cadena_markov(datos["secuencia"], orden=2)
                nueva[pista] = {
                    "instrumento": None,
                    "secuencia": generar_nueva_obra(modelo, len(datos["secuencia"]), 1.8, orden=2),
                }
            sintetizar_audio_directo(nueva, salida)
            memoria = nueva

        audio, sr = sf.read(salida)
        assert len(audio) > sr, "audio vacio"
        assert np.max(np.abs(audio)) <= 1.0, "audio fuera de rango"
        assert np.max(np.abs(audio)) > 0.05, "audio mudo"


if __name__ == "__main__":
    print("\n=== Bateria de regresion: La Cascada de Asimov ===\n")
    print(f"{TOTAL - len(FALLOS)}/{TOTAL} tests superados")
    if FALLOS:
        print("\nFALLOS:")
        for nombre, exc in FALLOS:
            print(f"  - {nombre}: {exc}")
        sys.exit(1)
    print("Todo correcto.")