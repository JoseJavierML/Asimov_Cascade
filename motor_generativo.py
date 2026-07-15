import random
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")



def construir_cadena_markov(secuencia: list, orden: int = 2) -> dict:
    def _normalizar_nota(nota):
        if nota in ("REST", ""):
            return "REST"
        if isinstance(nota, (tuple, list)):
            notas_normalizadas = []
            for elemento in nota:
                nota_normalizada = _normalizar_nota(elemento)
                if nota_normalizada == "REST":
                    continue
                if isinstance(nota_normalizada, tuple):
                    notas_normalizadas.extend(nota_normalizada)
                else:
                    notas_normalizadas.append(str(nota_normalizada))
            if not notas_normalizadas:
                return "REST"
            return tuple(dict.fromkeys(notas_normalizadas))
        if isinstance(nota, str) and nota.startswith("Acorde_"):
            partes = [parte for parte in nota.split("_", 1)[1].split(".") if parte]
            if not partes:
                return "REST"
            return tuple(partes)
        return (str(nota),)

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
        clave     = tuple(estados[i : i + orden_real])
        siguiente = estados[i + orden_real]
        transiciones.setdefault(clave, []).append(siguiente)

    return {
        "transiciones": transiciones,
        "estados": estados,
        "orden": orden_real,
        "escala_base": _escala_desde_secuencia(secuencia),
    }


def _notas_desde_evento(nota_str) -> list[int]:
    import librosa

    if nota_str in ("REST", ""):
        return []

    if isinstance(nota_str, (tuple, list)):
        notas_midi = []
        for elemento in nota_str:
            notas_midi.extend(_notas_desde_evento(elemento))
        return notas_midi

    if nota_str.startswith("Acorde_"):
        partes = nota_str.split("_", 1)[1].split(".")
        notas_midi = []
        for parte in partes:
            if not parte.isdigit():
                continue
            try:
                notas_midi.append(int(parte))
            except Exception:
                continue
        return notas_midi

    try:
        return [int(round(float(librosa.note_to_midi(str(nota_str)))))]
    except Exception:
        return []


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

    # Exponential easing: very flat at the start, then steep near the end.
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
    for estado in candidatos:
        nota = estado[0] if isinstance(estado, (list, tuple)) and len(estado) >= 1 else estado
        conteo = transiciones_actuales.count(estado) if transiciones_actuales else 1
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


def generar_nueva_obra(modelo: dict, longitud_base: int, temperatura: float, orden: int = 2) -> list:
    transiciones = modelo["transiciones"]
    estados      = modelo.get("estados", [])
    orden_real   = modelo["orden"]

    if not transiciones:
        return []

    todas_estados = list(dict.fromkeys(estados))

    variacion     = max(1, int(longitud_base * 0.1))
    longitud_final = random.randint(
        max(orden_real + 1, longitud_base - variacion),
        longitud_base + variacion,
    )

    clave_inicial = random.choice(list(transiciones.keys()))
    ventana = list(clave_inicial)
    nueva_sec = list(ventana)

    for _ in range(longitud_final - orden_real):
        clave_actual = tuple(ventana)

        estado_sig = _elegir_estado_por_temperatura(modelo, clave_actual, todas_estados, temperatura)
        nueva_sec.append(estado_sig)
        ventana = ventana[1:] + [estado_sig]

    return nueva_sec


def _dur_para(nota: str, duraciones: dict) -> float:
    if nota in duraciones and duraciones[nota]:
        return float(np.median(duraciones[nota]))
    todas = [d for ds in duraciones.values() for d in ds]
    return float(np.median(todas)) if todas else 0.4


def _vel_para(nota: str, velocidades: dict) -> float:
    if nota in velocidades and velocidades[nota]:
        return float(np.clip(np.median(velocidades[nota]), 0.0, 1.0))
    todas = [v for vs in velocidades.values() for v in vs]
    return float(np.clip(np.median(todas), 0.0, 1.0)) if todas else 1.0



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

def _perfil_instrumento(instrumento) -> dict:
    nombre = ""
    try:
        nombre = (instrumento.instrumentName or "").lower()
    except Exception:
        pass
    if "piano" in nombre or "keyb" in nombre:
        return _PERFILES["piano"]
    if "string" in nombre or "violin" in nombre or "cello" in nombre:
        return _PERFILES["strings"]
    if "organ" in nombre or "harm" in nombre:
        return _PERFILES["organ"]
    return _PERFILES["default"]


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
    attack = max(1, int(sr * float(perfil.get("attack", 0.008))))
    release = max(1, int(sr * float(perfil.get("release", 0.16))))
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


def _nota_a_freq(nota_str):
    import librosa
    if nota_str in ("REST", ""):
        return None
    try:
        if isinstance(nota_str, (tuple, list)):
            freqs = []
            for elemento in nota_str:
                freq_elemento = _nota_a_freq(elemento)
                if isinstance(freq_elemento, list):
                    freqs.extend(freq_elemento)
                elif freq_elemento is not None:
                    freqs.append(freq_elemento)
            return freqs or None
        if nota_str.startswith("Acorde_"):
            nums = [int(x) for x in nota_str.split("_")[1].split(".") if x.isdigit()]
            if not nums:
                return None
            freq = float(librosa.midi_to_hz(min(nums)))
        else:
            freq = float(librosa.note_to_hz(nota_str))
        return freq if 20 < freq < 20000 else None
    except Exception:
        return None


def sintetizar_audio_directo(
    diccionario_pistas: dict,
    nombre_salida: str,
    sr: int = 44100,
    amplitud_pista: float = 0.35,
) -> bool:
    import soundfile as sf

    pistas_audio = []
    errores_nota = 0
    notas_ok     = 0

    for nombre_pista, datos in diccionario_pistas.items():
        perfil     = _perfil_instrumento(datos.get("instrumento"))
        audio_pista = []

        for evento in datos["secuencia"]:
            if isinstance(evento, (list, tuple)) and len(evento) >= 2:
                nota_str, duracion = evento[0], evento[1]
                velocidad = float(evento[2]) if len(evento) >= 3 else 1.0
            else:
                errores_nota += 1
                continue

            if not isinstance(nota_str, (str, tuple, list)):
                errores_nota += 1
                continue

            duracion = max(0.05, float(duracion))
            velocidad = float(np.clip(velocidad, 0.0, 1.0))

            if nota_str == "REST":
                audio_pista.append(np.zeros(int(sr * duracion), dtype=np.float32))
                continue

            freq = _nota_a_freq(nota_str)
            if freq is None:
                errores_nota += 1
                continue

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
                audio_pista.append(_sintetizar_nota(freq, duracion, sr, amplitud_pista * velocidad, perfil))
            notas_ok += 1

        if audio_pista:
            pistas_audio.append(np.concatenate(audio_pista))

    if errores_nota > 0:
        print(f"  [Síntesis] {errores_nota} eventos ignorados (formato inválido o fuera de rango)")

    if not pistas_audio:
        raise RuntimeError(
            f"No se pudo sintetizar audio: 0 notas válidas procesadas "
            f"({errores_nota} eventos con errores). "
            "Revisa que la extracción musical produjo eventos con formato (nota_str, duracion)."
        )

    max_len = max(len(p) for p in pistas_audio)
    mezcla  = np.zeros(max_len, dtype=np.float32)
    for p in pistas_audio:
        mezcla[:len(p)] += p

    pico = np.max(np.abs(mezcla))
    if pico > 0:
        mezcla = mezcla / pico * 0.92

    sf.write(nombre_salida, mezcla, sr, subtype="PCM_16")
    print(f"  [Síntesis] ✓ {nombre_salida}  ({len(mezcla)/sr:.1f}s, {notas_ok} notas)")
    return True



if __name__ == "__main__":
    from memoria_base import extraer_adn_musical

    archivo_input    = "mi_obra.wav"
    orden_markov     = 2
    num_fases        = 15
    temperatura_maxima = 1.80

    if not os.path.exists(archivo_input):
        print(f"Error: no se encuentra '{archivo_input}'")
    else:
        memoria_actual = extraer_adn_musical(archivo_input)
        for datos in memoria_actual.values():
            datos["escala_base"] = _escala_desde_secuencia(datos["secuencia"])
        print("\n=== INICIANDO CASCADA DE ASIMOV ===\n")

        for f in range(1, num_fases + 1):
            temperatura = _temperatura_por_fase(f, num_fases, temperatura_maxima)
            print(f"Fase {f:02d}/{num_fases}  Temperatura: {temperatura:.2f}")

            nueva_memoria = {}
            for pista, datos in memoria_actual.items():
                modelo    = construir_cadena_markov(datos["secuencia"], orden=orden_markov)
                modelo["escala_base"] = datos.get("escala_base") or _escala_desde_secuencia(datos["secuencia"])
                nueva_sec = generar_nueva_obra(modelo, len(datos["secuencia"]), temperatura, orden=orden_markov)
                nueva_memoria[pista] = {
                    "instrumento": datos["instrumento"],
                    "secuencia": nueva_sec,
                    "escala_base": modelo["escala_base"],
                }

            sintetizar_audio_directo(nueva_memoria, f"Fase_{f:02d}_Audio.wav")
            memoria_actual = nueva_memoria

        print("\n=== PROCESO COMPLETADO ===")