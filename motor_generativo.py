import random
import os
import sys
import warnings
from collections import Counter
from functools import lru_cache

import numpy as np

warnings.filterwarnings("ignore")


def _forzar_utf8_consola() -> None:
    """Evita UnicodeEncodeError en consolas Windows (cp1252 por defecto)."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_forzar_utf8_consola()


@lru_cache(maxsize=8192)
def _midi_desde_token(token: str):
    """Resuelve un token suelto a MIDI: '60', 'C4', 'Bb3' o '60.5'. None si no aplica.

    Memorizado: _a_midi se llama por cada candidato y por cada paso del paseo,
    y los mismos nombres de nota se repiten miles de veces en una obra larga.
    """
    import librosa

    texto = token.strip()
    if not texto:
        return None

    if texto.isdigit() or (texto.startswith("-") and texto[1:].isdigit()):
        return int(texto)

    for candidato in (texto, texto.replace("-", "b")):
        try:
            return int(round(float(librosa.note_to_midi(candidato))))
        except Exception:
            continue
    return None


def _a_midi(nota) -> tuple:
    """Normaliza CUALQUIER representacion de nota a tupla de enteros MIDI.

    Acepta y unifica todas las formas que circulan por el motor:
        "REST" / "" / None   -> ()
        "Acorde_60.64.67"    -> (60, 64, 67)
        ("60", "64", "67")   -> (60, 64, 67)
        "C4" / "Bb3"         -> (60,)
        ("C4", "E4")         -> (60, 64)
        60 / 60.0            -> (60,)

    Motivo (fix v4.0): existian dos codificaciones incompatibles para el mismo
    acorde. "Acorde_60.64.67" funcionaba, pero la normalizacion de Markov la
    reconvertia a ('60','64','67'), y el sintetizador intentaba resolver cada
    elemento con librosa.note_to_hz('60'), que falla porque '60' no es un
    nombre de nota. Todos los acordes MIDI se descartaban en silencio (57 de 65
    eventos en la regresion) y la escala tonal quedaba vacia, perdiendo el ancla
    armonica del Softmax.
    """
    if nota is None or isinstance(nota, (bool, np.bool_)):
        return ()

    if isinstance(nota, (int, float, np.integer, np.floating)):
        valor = int(round(float(nota)))
        return (valor,) if 0 <= valor <= 127 else ()

    if isinstance(nota, (tuple, list, set)):
        unicos: list = []
        for elemento in nota:
            for midi in _a_midi(elemento):
                if midi not in unicos:
                    unicos.append(midi)
        return tuple(unicos)

    if not isinstance(nota, str):
        return ()

    texto = nota.strip()
    if not texto or texto.upper() in ("REST", "SILENCIO", "NONE"):
        return ()

    if texto.upper().startswith("ACORDE_"):
        cuerpo = texto.split("_", 1)[1].replace(",", ".")
        candidatos = cuerpo.split(".")
    else:
        candidatos = [texto]

    unicos = []
    for candidato in candidatos:
        midi = _midi_desde_token(candidato)
        if midi is not None and 0 <= midi <= 127 and midi not in unicos:
            unicos.append(midi)
    return tuple(unicos)


def construir_cadena_markov(secuencia: list, orden: int = 2) -> dict:
    def _normalizar_nota(nota):
        if isinstance(nota, str) and nota.strip().upper() in ("REST", "SILENCIO"):
            return "REST"
        midis = _a_midi(nota)
        return midis if midis else "REST"

    def _bin_velocidad(velocidad: float, es_descanso: bool = False) -> float:
        if es_descanso:
            return 0.0
        velocidad = float(np.clip(velocidad, 0.0, 1.0))
        if velocidad <= 0.0:
            return 0.2
        tier = int(np.clip(np.ceil(velocidad * 5.0), 1, 5))
        return tier / 5.0

    estados = []
    for evento in secuencia:
        if not isinstance(evento, (list, tuple)) or len(evento) < 2:
            continue

        nota = _normalizar_nota(evento[0])
        duracion = round(float(evento[1]), 6)
        velocidad = float(evento[2]) if len(evento) >= 3 else 1.0
        estado = (nota, duracion, _bin_velocidad(velocidad, nota == "REST"))
        estados.append(estado)

    n = len(estados)
    orden_real = min(orden, max(1, n - 1))

    transiciones: dict = {}
    for i in range(n - orden_real):
        clave = tuple(estados[i : i + orden_real])
        siguiente = estados[i + orden_real]
        transiciones.setdefault(clave, []).append(siguiente)

    return {
        "transiciones": transiciones,
        "estados": estados,
        "orden": orden_real,
        "escala_base": _escala_desde_secuencia(secuencia),
    }


def _notas_desde_evento(nota_str) -> list:
    """MIDI ints de cualquier representacion de nota. Delegado en el parser canonico."""
    return list(_a_midi(nota_str))


def _escala_desde_secuencia(secuencia: list) -> set[int]:
    escala = set()
    for evento in secuencia:
        if not isinstance(evento, (list, tuple)) or len(evento) < 2:
            continue
        nota = evento[0]
        for midi in _notas_desde_evento(nota):
            escala.add(midi % 12)
    return escala


def _afinidad_tonal(nota_str: str, escala_base: set[int]) -> float:
    if not escala_base:
        return 0.0

    notas_midi = _notas_desde_evento(nota_str)
    if not notas_midi:
        return 0.0

    coincidencias = sum(1 for midi in notas_midi if (midi % 12) in escala_base)
    return coincidencias / len(notas_midi)


def _softmax(logits: list[float] | np.ndarray, temperatura: float) -> np.ndarray:
    valores = np.asarray(logits, dtype=float)
    temp = max(float(temperatura), 1e-3)
    valores = valores / temp
    valores = valores - np.max(valores)
    exp = np.exp(valores)
    suma = float(np.sum(exp))
    if suma <= 0:
        return np.ones_like(exp) / len(exp)
    return exp / suma


def _temperatura_por_fase(
    fase: int,
    total_fases: int,
    temperatura_maxima: float,
    temperatura_minima: float = 0.10,
) -> float:
    total_fases = max(1, int(total_fases))
    fase = max(1, int(fase))

    if total_fases == 1:
        progreso = 1.0
    else:
        progreso = (fase - 1) / max(1, total_fases - 1)

    progreso = float(np.clip(progreso, 0.0, 1.0))

    curvatura = 5.0
    exponente = np.clip(curvatura * progreso, 0.0, 50.0)
    denom = np.expm1(curvatura)
    if abs(denom) < 1e-12:
        crecimiento = progreso
    else:
        crecimiento = np.expm1(exponente) / denom

    temperatura = temperatura_minima + (temperatura_maxima - temperatura_minima) * crecimiento
    return float(min(temperatura_maxima, max(temperatura_minima, temperatura)))


def _elegir_estado_por_temperatura(
    modelo: dict,
    clave_actual: tuple,
    todos_estados: list,
    temperatura: float,
) -> tuple:
    transiciones = modelo["transiciones"]
    escala_base = set(modelo.get("escala_base") or [])

    candidatos = list(dict.fromkeys(transiciones.get(clave_actual, []) or todos_estados))
    if not candidatos:
        return random.choice(todos_estados)

    logits = []
    transiciones_actuales = transiciones.get(clave_actual, [])
    # Counter en vez de .count() por candidato: antes era O(n*m) en el bucle caliente.
    frecuencias = Counter(transiciones_actuales) if transiciones_actuales else Counter()

    for estado in candidatos:
        nota = estado[0] if isinstance(estado, (list, tuple)) and len(estado) >= 1 else estado
        conteo = frecuencias.get(estado, 1) if frecuencias else 1
        afinidad = _afinidad_tonal(nota, escala_base)

        logit = np.log1p(conteo)
        if escala_base:
            if afinidad > 0:
                logit += 1.35 * afinidad / max(0.55, temperatura)
            else:
                logit -= 0.85 + 0.18 * max(0.0, temperatura - 1.0)

        logits.append(logit)

    pesos = _softmax(logits, temperatura)
    indice = np.random.choice(len(candidatos), p=pesos)
    return candidatos[int(indice)]


UMBRAL_LUCIDEZ = 0.35
"""Por debajo de esta temperatura el modelo todavia recuerda la obra.

Marca el limite entre la 'Lucidez Residual' y la 'Confusion' que describe el
README. Solo dentro de ese margen se fuerza un comportamiento fiel.
"""


def _peso_memoria(temperatura: float) -> float:
    """Probabilidad de conservar el recuerdo exacto en el siguiente evento.

    Una cadena de Markov pura no tiene memoria posicional: no sabe en que
    compas esta. En una pieza con motivos repetidos eso la encerraba en el
    primer ciclo (medido: de las 7 notas unicas del original se caian a 4,
    porque la clave que enlaza dos compas distintos se repetia 3 veces y
    ganaba siempre el mismo salto). Aqui el modelo recuerda la secuencia que
    tiene en este instante, y la temperatura decide cuanto se aparta de ella.

    La curva se acelera al acercarse al umbral: al principio apenas se desvía y,
    al llegar a UMBRAL_LUCIDEZ, ya no recuerda nada y es Markov puro.
    """
    if temperatura >= UMBRAL_LUCIDEZ:
        return 0.0
    return float(1.0 - (temperatura / UMBRAL_LUCIDEZ) ** 2)


def _clave_inicial(transiciones: dict, temperatura: float):
    """Elige por donde arrancar el paseo de Markov.

    Antes se sorteara una clave al azar, de modo que la Fase 1 ya comenzaba en
    un punto cualquiera de la pieza: medido, solo sobrevivia el 12.5% de las
    notas originales, y ese ratio era identico en la fase 1 y en la 10. Sin un
    inicio reconocible no hay colapso, hay ruido desde la primera iteracion.

    Mientras el modelo conserva lucidez se arranca en la primera clave, que es
    el comienzo real de la obra, porque el diccionario se construye recorriendo
    la secuencia en orden.
    """
    claves = list(transiciones.keys())
    if not claves:
        return None
    if temperatura <= UMBRAL_LUCIDEZ:
        return claves[0]
    return random.choice(claves)


def _reencontrar_referencia(referencia: list, ventana: list) -> int:
    """Indice de la referencia al que volver tras haberse desviado.

    Busca donde encaja la ventana actual. Solo se ejecuta cuando el paseo se
    sale de la referencia, que es la excepcion, no la norma.
    """
    if not ventana:
        return 0

    objetivo = tuple(ventana)
    for pos in range(0, max(1, len(referencia) - len(objetivo) + 1)):
        if tuple(referencia[pos:pos + len(objetivo)]) == objetivo:
            return pos + len(objetivo)
    return len(referencia)


def generar_nueva_obra(modelo: dict, longitud_base: int, temperatura: float, orden: int = 2) -> list:
    transiciones = modelo["transiciones"]
    estados = modelo.get("estados", [])
    orden_real = modelo["orden"]

    if not transiciones:
        return []

    todas_estados = list(dict.fromkeys(estados))
    lucido = temperatura <= UMBRAL_LUCIDEZ

    if lucido:
        longitud_final = max(orden_real + 1, longitud_base)
    else:
        variacion = max(1, int(longitud_base * 0.1))
        longitud_final = random.randint(
            max(orden_real + 1, longitud_base - variacion),
            longitud_base + variacion,
        )

    inicial = _clave_inicial(transiciones, temperatura)
    ventana = list(inicial)
    nueva_sec = list(ventana)

    referencia = list(estados)
    indice_ref = min(orden_real, len(referencia))
    peso_memoria = _peso_memoria(temperatura)

    for _ in range(longitud_final - orden_real):
        clave_actual = tuple(ventana)

        sigue_recordando = (
            peso_memoria > 0.0
            and indice_ref < len(referencia)
            and random.random() < peso_memoria
        )

        if sigue_recordando:
            estado_sig = referencia[indice_ref]
            indice_ref += 1
        else:
            estado_sig = _elegir_estado_por_temperatura(modelo, clave_actual, todas_estados, temperatura)
            indice_ref = _reencontrar_referencia(referencia, ventana + [estado_sig])

        nueva_sec.append(estado_sig)
        ventana = ventana[1:] + [estado_sig]

    return nueva_sec


_PERFILES = {
    "piano": {
        "carrier_ratio": 1.0,
        "mod_ratio": 2.0,
        "mod_index": 5.2,
        "mod_decay": 4.4,
        "amp_decay": 2.7,
        "attack": 0.008,
        "release": 0.160,
        "noise_mix": 0.060,
        "body_mix": 0.30,
        "detune_cents": 4.0,
        "warmth": 0.22,
        "drive": 1.10,
        "resonances": [
            (1.00, 0.22, 4.4),
            (2.01, 0.11, 5.8),
            (3.02, 0.05, 7.8),
            (4.10, 0.03, 9.6),
        ],
    },
    "strings": {
        "carrier_ratio": 1.0,
        "mod_ratio": 1.5,
        "mod_index": 3.0,
        "mod_decay": 3.0,
        "amp_decay": 1.9,
        "attack": 0.012,
        "release": 0.240,
        "noise_mix": 0.035,
        "body_mix": 0.24,
        "detune_cents": 2.5,
        "warmth": 0.20,
        "drive": 1.05,
        "resonances": [
            (1.00, 0.20, 3.8),
            (2.00, 0.10, 5.2),
            (3.00, 0.05, 7.0),
            (4.01, 0.02, 8.5),
        ],
    },
    "organ": {
        "carrier_ratio": 1.0,
        "mod_ratio": 2.0,
        "mod_index": 1.3,
        "mod_decay": 1.2,
        "amp_decay": 0.9,
        "attack": 0.012,
        "release": 0.180,
        "noise_mix": 0.015,
        "body_mix": 0.16,
        "detune_cents": 1.2,
        "warmth": 0.28,
        "drive": 1.00,
        "resonances": [
            (1.00, 0.16, 2.8),
            (2.00, 0.08, 3.8),
            (3.00, 0.04, 5.5),
        ],
    },
    "default": {
        "carrier_ratio": 1.0,
        "mod_ratio": 2.0,
        "mod_index": 4.0,
        "mod_decay": 3.5,
        "amp_decay": 2.2,
        "attack": 0.010,
        "release": 0.180,
        "noise_mix": 0.040,
        "body_mix": 0.22,
        "detune_cents": 3.0,
        "warmth": 0.22,
        "drive": 1.08,
        "resonances": [
            (1.00, 0.18, 3.8),
            (2.00, 0.09, 5.0),
            (3.02, 0.04, 6.8),
        ],
    },
}


def _perfil_instrumento(nombre_pista, instrumento, modo: str = "") -> dict:
    """Perfil FM del sintetizador.

    Antes solo miraba instrumento.instrumentName, que al releer un MIDI es None,
    asi que los perfiles 'strings' y 'organ' eran inalcanzables: ningun modo
    devuelto por _modo_sintesis_pista los pedia jamas.
    """
    try:
        nombre_instrumento = (instrumento.instrumentName or "").lower()
    except Exception:
        nombre_instrumento = ""

    texto = f"{(nombre_pista or '')} {nombre_instrumento}".strip().lower()

    if modo == "strings" or any(k in texto for k in ("string", "violin", "viola", "cello", "viol")):
        return _PERFILES["strings"]
    if modo == "organ" or any(k in texto for k in ("organ", "harm", "accordion")):
        return _PERFILES["organ"]
    if modo == "piano" or any(k in texto for k in ("piano", "keyb", "grand", "rhodes")):
        return _PERFILES["piano"]
    return _PERFILES["default"]


def _modo_sintesis_pista(nombre_pista: str, instrumento) -> str:
    nombre = (nombre_pista or "").lower()
    instrumento_nombre = ""
    try:
        instrumento_nombre = (instrumento.instrumentName or "").lower()
    except Exception:
        pass

    texto = f"{nombre} {instrumento_nombre}".strip()
    if "drum" in texto or "perc" in texto or "kit" in texto:
        return "drums"
    if "bass" in texto or "sub" in texto:
        return "bass"
    if "guitar" in texto or "pluck" in texto:
        return "guitar"
    if "violin" in texto or "cello" in texto or "viola" in texto or "string" in texto:
        return "strings"
    if "organ" in texto or "harm" in texto:
        return "organ"
    if "piano" in texto or "keyb" in texto or nombre == "master":
        return "piano"
    if nombre == "other" or "pad" in texto or "bell" in texto or "synth" in texto:
        return "other"
    return "other"


def _sintetizar_nota(freq: float, duracion: float, sr: int, amplitud: float, perfil: dict) -> np.ndarray:
    from scipy.signal import lfilter

    n = max(1, int(sr * duracion))
    perfil = perfil or _PERFILES["default"]
    freq = float(np.clip(freq, 20.0, min(sr * 0.45, 12000.0)))

    t = np.arange(n, dtype=np.float32) / float(sr)
    seed = int((freq * 1000.0) + (duracion * 1_000_000.0)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    carrier_ratio = float(perfil.get("carrier_ratio", 1.0))
    mod_ratio = float(perfil.get("mod_ratio", 2.0))
    mod_index = float(perfil.get("mod_index", 4.0))
    mod_decay = float(perfil.get("mod_decay", 3.0))
    amp_decay = float(perfil.get("amp_decay", 2.0))
    # Las envolventes no pueden medir mas que la propia nota. Sin este min(),
    # cualquier evento corto (la extraccion de audio produce notas de 50 ms)
    # provoca un ValueError de difusion en np.linspace.
    attack = min(max(1, int(sr * float(perfil.get("attack", 0.008)))), n)
    release = min(max(1, int(sr * float(perfil.get("release", 0.16)))), n)
    noise_mix = float(perfil.get("noise_mix", 0.04))
    body_mix = float(perfil.get("body_mix", 0.22))
    detune_cents = float(perfil.get("detune_cents", 0.0))
    warmth = float(perfil.get("warmth", 0.22))
    drive = float(perfil.get("drive", 1.0))
    resonances = perfil.get("resonances", [])

    attack_env = np.ones(n, dtype=np.float32)
    if attack > 1:
        ataque = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
        ataque = np.clip(ataque, 0.0, 1.0)
        attack_env[:attack] = ataque

    release_env = np.ones(n, dtype=np.float32)
    if release < n:
        release_env[-release:] = np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)

    decay_rate = amp_decay / max(0.7, np.sqrt(freq / 220.0))
    amp_env = np.exp(-t * decay_rate).astype(np.float32)
    envelope = np.clip(attack_env * release_env * amp_env, 0.0, 1.0)

    mod_env = np.exp(-t * mod_decay).astype(np.float32)
    mod_index_t = mod_index * mod_env
    mod_index_t *= np.clip(1.10 - (freq / 9000.0), 0.35, 1.0)

    detune = (detune_cents / 1200.0) * (0.65 + 0.35 * np.sin(freq * 0.017))
    fase_car = rng.uniform(0.0, 2.0 * np.pi)
    fase_mod = rng.uniform(0.0, 2.0 * np.pi)

    modulador = np.sin(2.0 * np.pi * freq * mod_ratio * t + fase_mod)
    portadora = np.sin(
        2.0 * np.pi * freq * carrier_ratio * (1.0 + detune) * t
        + mod_index_t * modulador
        + fase_car
    )

    segunda_capa = np.sin(
        2.0 * np.pi * freq * (carrier_ratio * 2.0) * (1.0 - 0.5 * detune) * t
        + 0.55 * mod_index_t * np.sin(2.0 * np.pi * freq * (mod_ratio * 1.01) * t + 0.7 * fase_mod)
        + 0.43 * fase_car
    )

    senal = (0.76 * portadora) + (0.24 * segunda_capa)

    if noise_mix > 0:
        ataque_ruido = rng.normal(0.0, 1.0, n).astype(np.float32)
        ataque_ruido = lfilter([0.5, 0.5], [1.0], ataque_ruido).astype(np.float32)
        ataque_ruido *= np.exp(-t * 180.0).astype(np.float32)
        senal += noise_mix * ataque_ruido

    if resonances:
        cuerpo = np.zeros(n, dtype=np.float32)
        for ratio, amp, decay in resonances:
            fase = rng.uniform(0.0, 2.0 * np.pi)
            cuerpo += (
                float(amp)
                * np.sin(2.0 * np.pi * freq * float(ratio) * t + fase).astype(np.float32)
                * np.exp(-float(decay) * t).astype(np.float32)
            )
        senal += body_mix * cuerpo

    if n > 4:
        senal = lfilter([warmth], [1.0, warmth - 1.0], senal).astype(np.float32)

    senal = np.tanh(senal * drive).astype(np.float32)
    return (amplitud * senal * envelope).astype(np.float32)


def _sintetizar_subbass(freq: float, duracion: float, sr: int, amplitud: float) -> np.ndarray:
    from scipy.signal import lfilter

    n = max(1, int(sr * duracion))
    freq = float(np.clip(freq, 20.0, min(sr * 0.22, 220.0)))

    t = np.arange(n, dtype=np.float32) / float(sr)
    phase = float((freq * 997.0) % (2.0 * np.pi))

    sub = np.sin(2.0 * np.pi * freq * t + phase).astype(np.float32)
    square = np.sign(np.sin(2.0 * np.pi * freq * 0.5 * t + 0.73 * phase)).astype(np.float32)
    body = 0.72 * sub + 0.28 * square

    attack = min(max(1, int(sr * 0.006)), n)
    release = min(max(1, int(sr * min(0.20, max(0.04, duracion * 0.35)))), n)
    ataque = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
    liberacion = np.ones(n, dtype=np.float32)
    if release < n:
        liberacion[-release:] = np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)
    envolvente = np.ones(n, dtype=np.float32)
    envolvente[:attack] = ataque

    cuerpo = np.tanh(body * 1.4).astype(np.float32)
    cuerpo = lfilter([1.0], [1.0, -0.96], cuerpo).astype(np.float32)
    cuerpo = lfilter([0.22], [1.0, -0.78], cuerpo).astype(np.float32)
    cuerpo = np.tanh(cuerpo * 1.5).astype(np.float32)
    return (amplitud * cuerpo * envolvente * liberacion).astype(np.float32)


def _sintetizar_pad_bell(freq: float, duracion: float, sr: int, amplitud: float) -> np.ndarray:
    from scipy.signal import lfilter

    n = max(1, int(sr * duracion))
    freq = float(np.clip(freq, 20.0, min(sr * 0.35, 5000.0)))
    t = np.arange(n, dtype=np.float32) / float(sr)
    rng = np.random.default_rng(int(freq * 7919 + duracion * 1_000_000) & 0xFFFFFFFF)

    attack = min(max(1, int(sr * 0.12)), n)
    release = min(max(1, int(sr * max(0.50, min(2.50, duracion * 0.75)))), n)
    env = np.exp(-t * 0.65).astype(np.float32)
    env[:attack] *= np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
    if release < n:
        env[-release:] *= np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)

    fase1 = rng.uniform(0.0, 2.0 * np.pi)
    fase2 = rng.uniform(0.0, 2.0 * np.pi)
    base = np.sin(2.0 * np.pi * freq * t + fase1).astype(np.float32)
    overtone = 0.48 * np.sin(2.0 * np.pi * freq * 2.01 * t + fase2).astype(np.float32)
    shimmer = 0.16 * np.sin(2.0 * np.pi * freq * 3.98 * t + 0.5 * fase1).astype(np.float32)
    nube = base + overtone + shimmer

    ruido = rng.normal(0.0, 1.0, n).astype(np.float32)
    ruido = lfilter([0.25, 0.25], [1.0], ruido).astype(np.float32)
    ruido *= np.exp(-t * 10.0).astype(np.float32)

    senal = nube + 0.08 * ruido
    senal = lfilter([0.18], [1.0, -0.82], senal).astype(np.float32)
    senal = np.tanh(senal * 1.15).astype(np.float32)
    return (amplitud * senal * env).astype(np.float32)


def _sintetizar_guitarra(freq: float, duracion: float, sr: int, amplitud: float, velocidad: float) -> np.ndarray:
    from scipy.signal import sawtooth, lfilter

    n = max(1, int(sr * duracion))
    freq = float(np.clip(freq, 40.0, min(sr * 0.30, 4000.0)))
    t = np.arange(n, dtype=np.float32) / float(sr)
    rng = np.random.default_rng(int(freq * 1237 + duracion * 1_000_000) & 0xFFFFFFFF)

    attack = min(max(1, int(sr * 0.008)), n)
    release = min(max(1, int(sr * min(0.30, max(0.05, duracion * 0.45)))), n)
    env = np.exp(-t * (2.4 + 0.8 * (1.0 - float(np.clip(velocidad, 0.0, 1.0))))).astype(np.float32)
    env[:attack] *= np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
    if release < n:
        env[-release:] *= np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)

    pulse = sawtooth(2.0 * np.pi * freq * t, width=0.38).astype(np.float32)
    body = 0.75 * pulse + 0.18 * np.sin(2.0 * np.pi * freq * 2.0 * t).astype(np.float32)
    body += 0.05 * rng.normal(0.0, 1.0, n).astype(np.float32)
    body = lfilter([0.28], [1.0, -0.72], body).astype(np.float32)
    body = np.tanh(body * 1.25).astype(np.float32)
    return (amplitud * body * env).astype(np.float32)


def _sintetizar_drums(duracion: float, sr: int, amplitud: float, velocidad: float, nota_str=None) -> np.ndarray:
    import hashlib
    from scipy.signal import butter, sosfilt

    n = max(1, int(sr * duracion))
    token = f"{nota_str}|{duracion:.4f}|{float(velocidad):.3f}".encode("utf-8", errors="ignore")
    seed_base = int.from_bytes(hashlib.sha1(token).digest()[:8], "little")
    rng = np.random.default_rng(seed_base)

    t = np.arange(n, dtype=np.float32) / float(sr)
    velocidad = float(np.clip(velocidad, 0.0, 1.0))

    ruido = rng.normal(0.0, 1.0, n).astype(np.float32)
    if n > 4:
        sos = butter(3, [180.0 / (sr * 0.5), 7600.0 / (sr * 0.5)], btype="band", output="sos")
        ruido = sosfilt(sos, ruido).astype(np.float32)

    ataque = min(max(1, int(sr * 0.0015)), n)
    decaimiento = min(max(1, int(sr * max(0.03, min(0.30, duracion * 0.28)))), n)
    env = np.exp(-t * (18.0 + 18.0 * (1.0 - velocidad))).astype(np.float32)
    env[:ataque] *= np.linspace(0.0, 1.0, ataque, endpoint=False, dtype=np.float32)
    if decaimiento < n:
        env[-decaimiento:] *= np.linspace(1.0, 0.0, decaimiento, endpoint=True, dtype=np.float32)

    golpe = ruido * env
    golpe += 0.15 * np.sin(2.0 * np.pi * 160.0 * t).astype(np.float32) * env
    golpe = np.tanh(golpe * (1.0 + 0.9 * velocidad)).astype(np.float32)
    return (amplitud * golpe).astype(np.float32)


def _nota_a_freq(nota_str):
    """Frecuencia (Hz) de una nota o de un acorde.

    Devuelve None para silencios o notas irrecuperables, un float para
    monofonia y una lista de floats para acordes. Un acorde MIDI ahora suena
    como acorde (antes la rama polifonica era inalcanzable).
    """
    import librosa

    midis = _a_midi(nota_str)
    if not midis:
        return None

    frecuencias = []
    for midi in midis:
        try:
            freq = float(librosa.midi_to_hz(midi))
        except Exception:
            continue
        if 20.0 < freq < 20000.0:
            frecuencias.append(freq)

    if not frecuencias:
        return None
    return frecuencias[0] if len(frecuencias) == 1 else frecuencias


def _renderizar_pistas(
    diccionario_pistas: dict,
    sr: int = 44100,
    amplitud_pista: float = 0.35,
) -> np.ndarray:
    """Renderiza el diccionario de pistas a un unico buffer mono.

    Extraido de sintetizar_audio_directo para poder reutilizarlo sin escribir
    a disco: la cascada continua necesita los audios en memoria para fundirlos.
    """
    pistas_audio = []
    errores_nota = 0
    notas_ok = 0

    for nombre_pista, datos in diccionario_pistas.items():
        if isinstance(datos, dict):
            secuencia = datos.get("secuencia", [])
            instrumento = datos.get("instrumento")
            modo_explicito = datos.get("modo_sintesis")
        else:
            secuencia = datos
            instrumento = None
            modo_explicito = None

        perfil = _perfil_instrumento(nombre_pista, instrumento, modo_explicito or "")
        modo_sintesis = modo_explicito or _modo_sintesis_pista(nombre_pista, instrumento)
        audio_pista = []

        for evento in secuencia:
            if not isinstance(evento, (list, tuple)) or len(evento) < 2:
                errores_nota += 1
                continue

            nota_str = evento[0]
            duracion = max(0.05, float(evento[1]))
            velocidad = float(evento[2]) if len(evento) >= 3 else 1.0
            velocidad = float(np.clip(velocidad, 0.0, 1.0))

            if nota_str == "REST":
                audio_pista.append(np.zeros(int(sr * duracion), dtype=np.float32))
                continue

            freq = _nota_a_freq(nota_str)
            if freq is None:
                errores_nota += 1
                continue

            if modo_sintesis == "drums":
                audio_pista.append(_sintetizar_drums(duracion, sr, amplitud_pista * velocidad, velocidad, nota_str))
            elif modo_sintesis == "bass":
                if isinstance(freq, list):
                    if not freq:
                        errores_nota += 1
                        continue
                    freq = min(freq)
                audio_pista.append(_sintetizar_subbass(float(freq), duracion, sr, amplitud_pista * 1.15 * velocidad))
            elif modo_sintesis == "guitar":
                if isinstance(freq, list):
                    if not freq:
                        errores_nota += 1
                        continue
                    freq = min(freq)
                audio_pista.append(_sintetizar_guitarra(float(freq), duracion, sr, amplitud_pista * 0.95 * velocidad, velocidad))
            elif modo_sintesis == "other":
                if isinstance(freq, list):
                    if not freq:
                        errores_nota += 1
                        continue
                    voces = []
                    amplitud_voces = (amplitud_pista * 0.85 * velocidad) / max(1, len(freq))
                    for indice, freq_voz in enumerate(freq):
                        freq_detune = float(freq_voz) * (1.0 + 0.0007 * (indice - (len(freq) - 1) / 2.0))
                        voces.append(_sintetizar_pad_bell(freq_detune, duracion, sr, amplitud_voces))
                    bloque_mezclado = np.sum(voces, axis=0).astype(np.float32)
                    bloque_mezclado /= max(1, len(voces))
                    audio_pista.append(bloque_mezclado)
                else:
                    audio_pista.append(_sintetizar_pad_bell(float(freq), duracion, sr, amplitud_pista * 0.85 * velocidad))
            else:
                if isinstance(freq, list):
                    if not freq:
                        errores_nota += 1
                        continue
                    voces = []
                    amplitud_voces = (amplitud_pista * velocidad) / max(1, len(freq))
                    for indice, freq_voz in enumerate(freq):
                        freq_detune = float(freq_voz) * (1.0 + 0.0015 * (indice - (len(freq) - 1) / 2.0))
                        voces.append(_sintetizar_nota(freq_detune, duracion, sr, amplitud_voces, perfil))
                    bloque_mezclado = np.sum(voces, axis=0).astype(np.float32)
                    bloque_mezclado /= max(1, len(voces))
                    audio_pista.append(bloque_mezclado)
                else:
                    audio_pista.append(_sintetizar_nota(float(freq), duracion, sr, amplitud_pista * velocidad, perfil))

            notas_ok += 1

        if audio_pista:
            pistas_audio.append(np.concatenate(audio_pista))

    if errores_nota > 0:
        print(f"  [Sintesis] {errores_nota} eventos ignorados (formato invalido o fuera de rango)")

    if not pistas_audio:
        raise RuntimeError(
            f"No se pudo sintetizar audio: 0 notas válidas procesadas "
            f"({errores_nota} eventos con errores). "
            "Revisa que la extracción musical produjo eventos con formato (nota_str, duracion)."
        )

    max_len = max(len(p) for p in pistas_audio)
    mezcla = np.zeros(max_len, dtype=np.float32)
    for pista_audio in pistas_audio:
        mezcla[: len(pista_audio)] += pista_audio

    pico = float(np.max(np.abs(mezcla)))
    if pico > 1.0:
        mezcla = np.tanh(mezcla).astype(np.float32)

    pico = float(np.max(np.abs(mezcla)))
    if pico > 0:
        mezcla = (mezcla / pico) * 0.92

    return np.clip(mezcla, -0.98, 0.98).astype(np.float32), notas_ok


def sintetizar_audio_directo(
    diccionario_pistas: dict,
    nombre_salida: str,
    sr: int = 44100,
    amplitud_pista: float = 0.35,
) -> bool:
    import soundfile as sf

    mezcla, notas_ok = _renderizar_pistas(diccionario_pistas, sr, amplitud_pista)

    sf.write(nombre_salida, mezcla, sr, subtype="PCM_16")
    print(f"  [Sintesis] OK  {nombre_salida}  ({len(mezcla)/sr:.1f}s, {notas_ok} notas)")
    return True


def _fundir_buffers(buffers: list, sr: int, crossfade_s: float) -> np.ndarray:
    """Encadena buffers solapandolos con un fundido de igual energia.

    Cada fase es una version distinta de la misma obra, y elTitulo del proyecto
    habla de 'cascada'. Con archivos sueltos el oyente tiene que saltar de uno a
    otro a mano; aqui cada iteracion se derrama sobre la siguiente y se escucha
    la obra deshacerse en una sola toma continua.

    La curva es raiz cuadrada (ley de cosenos simplificada): con una mezcla
    lineal el centro del cruce se hunde, porque dos senos desfasados suman
    energia pero no amplitud.
    """
    buffers = [np.asarray(b, dtype=np.float32) for b in buffers if len(b) > 0]
    if not buffers:
        return np.zeros(0, dtype=np.float32)
    if len(buffers) == 1:
        return buffers[0]

    muestras = int(sr * max(0.0, float(crossfade_s)))
    if muestras > 0:
        muestras = min(muestras, min(len(b) for b in buffers))
    if muestras <= 0:
        return np.concatenate(buffers).astype(np.float32)

    curva = np.sqrt(np.linspace(0.0, 1.0, muestras, dtype=np.float32))

    resultado = buffers[0]
    for siguiente in buffers[1:]:
        cola = resultado[-muestras:]
        cabeza = siguiente[:muestras]
        cruce = (cola * (1.0 - curva) + cabeza * curva).astype(np.float32)
        resultado = np.concatenate([resultado[:-muestras], cruce, siguiente[muestras:]])

    return resultado.astype(np.float32)


def sintetizar_cascada(
    fases: list,
    nombre_salida: str,
    sr: int = 44100,
    amplitud_pista: float = 0.35,
    crossfade_s: float = 1.5,
) -> bool:
    """Funde todas las fases en un unico WAV continuo.

    fases: lista de diccionarios de pistas, en orden de degradacion.
    Devuelve False si no hay ninguna fase utilizable.
    """
    import soundfile as sf

    fases_validas = [f for f in fases if f]
    if not fases_validas:
        print("  [Cascada] No hay fases validas para fundir.")
        return False

    buffers = []
    for indice, fase in enumerate(fases_validas, start=1):
        mezcla, _notas = _renderizar_pistas(fase, sr, amplitud_pista)
        buffers.append(mezcla)

    cascada = _fundir_buffers(buffers, sr, crossfade_s)
    if cascada.size == 0:
        print("  [Cascada] El fundido quedo vacio.")
        return False

    pico = float(np.max(np.abs(cascada)))
    if pico > 0:
        cascada = (cascada / pico) * 0.92
    cascada = np.clip(cascada, -0.98, 0.98).astype(np.float32)

    sf.write(nombre_salida, cascada, sr, subtype="PCM_16")
    solapes = max(0, len(fases_validas) - 1)
    print(f"  [Cascada] OK  {nombre_salida}  "
          f"({len(cascada)/sr:.1f}s, {len(fases_validas)} fases fundidas, "
          f"{crossfade_s:.2f}s de solape cada una, pico {pico:.2f})")
    return True


if __name__ == "__main__":
    from memoria_base import extraer_adn_musical

    archivo_input = "mi_obra.wav"
    orden_markov = 2
    num_fases = 15
    temperatura_maxima = 1.80
    fundir_cascada = True
    crossfade_s = 1.5

    if not os.path.exists(archivo_input):
        print(f"Error: no se encuentra '{archivo_input}'")
    else:
        memoria_actual = extraer_adn_musical(archivo_input)
        for datos in memoria_actual.values():
            datos["escala_base"] = _escala_desde_secuencia(datos["secuencia"])
        print("\n=== INICIANDO CASCADA DE ASIMOV ===\n")

        fases: list = []

        for f in range(1, num_fases + 1):
            temperatura = _temperatura_por_fase(f, num_fases, temperatura_maxima)
            print(f"Fase {f:02d}/{num_fases}  Temperatura: {temperatura:.2f}")

            nueva_memoria = {}
            for pista, datos in memoria_actual.items():
                modelo = construir_cadena_markov(datos["secuencia"], orden=orden_markov)
                modelo["escala_base"] = datos.get("escala_base") or _escala_desde_secuencia(datos["secuencia"])
                nueva_sec = generar_nueva_obra(modelo, len(datos["secuencia"]), temperatura, orden=orden_markov)
                nueva_memoria[pista] = {
                    "instrumento": datos["instrumento"],
                    "secuencia": nueva_sec,
                    "escala_base": modelo["escala_base"],
                    "bpm": datos.get("bpm"),
                    "modo_sintesis": datos.get("modo_sintesis"),
                }

            sintetizar_audio_directo(nueva_memoria, f"Fase_{f:02d}_Audio.wav")
            fases.append(nueva_memoria)
            memoria_actual = nueva_memoria

        if fundir_cascada:
            sintetizar_cascada(fases, "Cascada_Continua.wav", crossfade_s=crossfade_s)

        print("\n=== PROCESO COMPLETADO ===")