"""Bateria de regresion de La Cascada de Asimov.

Ejecucion sin dependencias externas (sin pytest):

    python tests/test_cascada.py

Cada test documenta un fallo concreto que ya ocurria y que no debe volver.
"""
import os
import sys
import tempfile

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


# --------------------------------------------------------------------------
# Bloque 2: reloj temporal. El MIDI guardaba duraciones en negras, y el
# sintetizador las trataba como segundos, así que todo sonaba a 60 BPM.
# --------------------------------------------------------------------------

def _midi_de_prueba(ruta, bpm, negras=4, cambio_tempo=None):
    import music21 as m21

    score = m21.stream.Score()
    parte = m21.stream.Part()
    parte.append(m21.instrument.Piano())
    for i in range(negras):
        nota = m21.note.Note("C4")
        nota.quarterLength = 1.0
        nota.volume.velocity = 64 + i * 15
        parte.append(nota)
    score.insert(0, parte)
    score.insert(0, m21.tempo.MetronomeMark(number=bpm))
    if cambio_tempo:
        score.insert(0, m21.tempo.MetronomeMark(number=cambio_tempo[1]))
        score.getElementsByClass(m21.tempo.MetronomeMark)[1].offset = cambio_tempo[0]
    score.write("midi", fp=ruta)
    return ruta


@test
def test_duracion_midi_respeta_el_tempo():
    """Bug C: 4 negras a 120 BPM duran 2.0 s, no 4.0 s."""
    from memoria_base import extraer_adn_musical

    with tempfile.TemporaryDirectory() as tmp:
        ruta = _midi_de_prueba(os.path.join(tmp, "t.mid"), bpm=120, negras=4)
        memoria = extraer_adn_musical(ruta)
        total = sum(float(d) for pista in memoria.values() for _n, d, _v in pista["secuencia"])
        assert abs(total - 2.0) < 0.05, f"esperaba 2.0s, obtuve {total:.3f}s"


@test
def test_tempos_distintos_dan_duraciones_distintas():
    from memoria_base import extraer_adn_musical

    with tempfile.TemporaryDirectory() as tmp:
        lentas = extraer_adn_musical(_midi_de_prueba(os.path.join(tmp, "lenta.mid"), bpm=60, negras=4))
        rapidas = extraer_adn_musical(_midi_de_prueba(os.path.join(tmp, "rapida.mid"), bpm=180, negras=4))
        t_lenta = sum(float(d) for p in lentas.values() for _n, d, _v in p["secuencia"])
        t_rapida = sum(float(d) for p in rapidas.values() for _n, d, _v in p["secuencia"])
        assert abs(t_lenta - 4.0) < 0.05, t_lenta
        assert abs(t_rapida - 1.333) < 0.05, t_rapida


@test
def test_bpm_queda_registrado_en_la_memoria():
    from memoria_base import extraer_adn_musical

    with tempfile.TemporaryDirectory() as tmp:
        memoria = extraer_adn_musical(_midi_de_prueba(os.path.join(tmp, "b.mid"), bpm=140, negras=2))
        for pista in memoria.values():
            assert abs(pista["bpm"] - 140.0) < 0.5, pista.get("bpm")


@test
def test_cambio_de_tempo_se_integra_en_la_duracion():
    """Una nota que cruza un accelerando debe integrar ambos tramos."""
    from memoria_base import _duracion_en_segundos

    mapa = [(0.0, 0.5), (4.0, 0.25)]  # 120 BPM, luego 240 BPM
    total = _duracion_en_segundos(mapa, 2.0, 4.0)  # 2 negras a 120 + 2 a 240
    assert abs(total - 1.5) < 1e-6, total


@test
def test_velocidad_midi_se_recupera():
    """Bug 4: la velocidad era siempre 1.0 y las dinamicas eran ficticias."""
    from memoria_base import extraer_adn_musical

    with tempfile.TemporaryDirectory() as tmp:
        ruta = _midi_de_prueba(os.path.join(tmp, "v.mid"), bpm=120, negras=4)
        memoria = extraer_adn_musical(ruta)
        velocidades = [v for pista in memoria.values() for _n, _d, v in pista["secuencia"] if v > 0]
        assert len(set(velocidades)) >= 3, velocidades
        assert all(0.0 < v <= 1.0 for v in velocidades), velocidades


@test
def test_sintesis_respeta_el_tempo_real():
    """El audio generado debe durar lo que dura la obra original."""
    import soundfile as sf
    from memoria_base import extraer_adn_musical
    from motor_generativo import sintetizar_audio_directo

    with tempfile.TemporaryDirectory() as tmp:
        ruta = _midi_de_prueba(os.path.join(tmp, "x.mid"), bpm=120, negras=8)
        memoria = extraer_adn_musical(ruta)
        esperado = sum(float(d) for p in memoria.values() for _n, d, _v in p["secuencia"])

        salida = os.path.join(tmp, "salida.wav")
        sintetizar_audio_directo(memoria, salida)
        audio, sr = sf.read(salida)
        assert abs(len(audio) / sr - esperado) < 0.1, f"audio {len(audio)/sr:.2f}s vs esperado {esperado:.2f}s"


# --------------------------------------------------------------------------
# Bloque P1: fidelidad de la Fase 1. Antes arrancaba en un punto aleatorio y
# solo sobrevivia el 12.5% de la obra, igual en la fase 1 que en la 10.
# --------------------------------------------------------------------------

@test
def test_peso_memoria_decae_con_la_temperatura():
    from motor_generativo import UMBRAL_LUCIDEZ, _peso_memoria

    assert _peso_memoria(0.0) == 1.0
    assert _peso_memoria(UMBRAL_LUCIDEZ) == 0.0
    assert _peso_memoria(9.99) == 0.0
    medio = _peso_memoria(UMBRAL_LUCIDEZ / 2)
    assert 0.5 < medio < 1.0, medio
    assert _peso_memoria(0.10) > _peso_memoria(0.25)


@test
def test_fase_lucida_reproduce_la_obra():
    """Con memoria casi integra, la Fase 1 debe devolver la obra tal cual."""
    from motor_generativo import construir_cadena_markov, generar_nueva_obra

    original = [((60,), 0.5, 0.6), ((62,), 0.5, 0.8), ((64,), 0.5, 0.8), ((65,), 0.5, 0.6),
                ((67,), 0.5, 0.8), ((64,), 0.5, 0.8), ((62,), 0.5, 0.6), ((60,), 0.5, 0.8)] * 6
    modelo = construir_cadena_markov(original, orden=2)
    fase1 = generar_nueva_obra(modelo, len(original), 0.10, orden=2)

    assert len(fase1) == len(original), (len(fase1), len(original))
    assert fase1[:len(fase1)] == original[:len(original)][:len(fase1)]


@test
def test_la_fase_lucida_aun_recuerda_la_obra():
    """El olvido debe ser gradual: la primera fase conserva la mayor parte."""
    import random as _random
    import numpy as _np
    from motor_generativo import _temperatura_por_fase

    original = [((55 + (i * 2) % 24,), 0.25, 0.6) for i in range(60)]
    _random.seed(11)
    _np.random.seed(11)

    primera = generar_nueva_obra(
        construir_cadena_markov(original, orden=2),
        len(original),
        _temperatura_por_fase(1, 10, 1.80),
        orden=2,
    )
    a, b = [e[0] for e in original], [e[0] for e in primera]
    n = min(len(a), len(b))
    coincidencia = sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b))
    assert coincidencia >= 0.5, f"la fase lucida deberia recordar la obra, solo {coincidencia:.0%}"


@test
def test_alta_temperatura_es_markov_puro():
    """Pasado el umbral, la memoria se apaga y no queda ancla posicional."""
    from motor_generativo import UMBRAL_LUCIDEZ

    original = [((60,), 0.5, 0.6), ((62,), 0.5, 0.8), ((64,), 0.5, 0.8)] * 20
    modelo = construir_cadena_markov(original, orden=2)
    resultados = set()
    for _ in range(5):
        sec = generar_nueva_obra(modelo, len(original), UMBRAL_LUCIDEZ + 0.5, orden=2)
        resultados.add(tuple(e[0] for e in sec))
    assert len(resultados) > 1, "a alta temperatura cada pasada deberia diferir"


@test
def test_cascada_completa_no_revienta():
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