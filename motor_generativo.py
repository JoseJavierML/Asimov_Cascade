import random
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")



def construir_cadena_markov(secuencia: list, orden: int = 2) -> dict:
  
    notas    = [e[0] for e in secuencia]
    durs     = [e[1] for e in secuencia]
    n        = len(notas)

    orden_real = min(orden, max(1, n - 1))

    transiciones: dict = {}
    for i in range(n - orden_real):
        clave     = tuple(notas[i : i + orden_real])
        siguiente = notas[i + orden_real]
        transiciones.setdefault(clave, []).append(siguiente)

    duraciones: dict = {}
    for nota, dur in zip(notas, durs):
        duraciones.setdefault(nota, []).append(dur)

    return {
        "transiciones": transiciones,
        "duraciones": duraciones,
        "orden": orden_real,
        "escala_base": _escala_desde_secuencia(secuencia),
    }


def _notas_desde_evento(nota_str: str) -> list[int]:
    import librosa

    if not isinstance(nota_str, str) or nota_str in ("REST", ""):
        return []

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
        return [int(round(float(librosa.note_to_midi(nota_str))))]
    except Exception:
        return []


def _escala_desde_secuencia(secuencia: list) -> set[int]:
    escala = set()
    for nota, _dur in secuencia:
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


def _temperatura_por_fase(progreso: float, temperatura_maxima: float, temperatura_minima: float = 0.18) -> float:
    progreso = float(np.clip(progreso, 0.0, 1.0))
    curvatura = 4.0
    crecimiento = np.expm1(curvatura * progreso) / np.expm1(curvatura)
    return float(temperatura_minima + (temperatura_maxima - temperatura_minima) * crecimiento)


def _elegir_nota_por_temperatura(
    modelo: dict,
    clave_actual: tuple,
    todas_notas: list[str],
    temperatura: float,
) -> str:
    transiciones = modelo["transiciones"]
    escala_base = set(modelo.get("escala_base") or [])

    candidatos = list(dict.fromkeys(transiciones.get(clave_actual, []) or todas_notas))
    if not candidatos:
        return random.choice(todas_notas)

    logits = []
    transiciones_actuales = transiciones.get(clave_actual, [])
    for nota in candidatos:
        conteo = transiciones_actuales.count(nota) if transiciones_actuales else 1
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
    duraciones   = modelo["duraciones"]
    orden_real   = modelo["orden"]

    if not transiciones:
        return []

    todas_notas = list(duraciones.keys())   

    variacion     = max(1, int(longitud_base * 0.1))
    longitud_final = random.randint(
        max(orden_real + 1, longitud_base - variacion),
        longitud_base + variacion,
    )

    clave_inicial = random.choice(list(transiciones.keys()))
    ventana       = list(clave_inicial)          
    nueva_sec     = [(n, _dur_para(n, duraciones)) for n in ventana]

    for _ in range(longitud_final - orden_real):
        clave_actual = tuple(ventana)

        nota_sig = _elegir_nota_por_temperatura(modelo, clave_actual, todas_notas, temperatura)

        prob_desvio_duracion = min(0.75, 0.08 + (temperatura * 0.16))
        if random.random() < prob_desvio_duracion:
            todas_durs = [d for ds in duraciones.values() for d in ds]
            dur_sig    = random.choice(todas_durs) if todas_durs else 0.4
        else:
            dur_base = _dur_para(nota_sig, duraciones)
            ruido    = np.random.normal(0, max(0.04, temperatura * 0.08))
            dur_sig  = max(0.05, round(dur_base * (1 + ruido), 3))

        nueva_sec.append((nota_sig, dur_sig))
        ventana = ventana[1:] + [nota_sig]

    return nueva_sec


def _dur_para(nota: str, duraciones: dict) -> float:
    if nota in duraciones and duraciones[nota]:
        return float(np.median(duraciones[nota]))
    todas = [d for ds in duraciones.values() for d in ds]
    return float(np.median(todas)) if todas else 0.4



_PERFILES = {
    "piano":   [1.00, 0.60, 0.30, 0.18, 0.10, 0.06, 0.03, 0.01],
    "strings": [1.00, 0.80, 0.60, 0.40, 0.25, 0.15, 0.08, 0.04],
    "organ":   [1.00, 1.00, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
    "default": [1.00, 0.50, 0.25, 0.12, 0.06, 0.03, 0.01, 0.00],
}

def _perfil_instrumento(instrumento) -> list:
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


def _sintetizar_nota(freq: float, duracion: float, sr: int, amplitud: float, perfil: list) -> np.ndarray:
    n = max(1, int(sr * duracion))
    t = np.linspace(0, duracion, n, endpoint=False)
    onda = np.zeros(n)
    for k, amp_p in enumerate(perfil, start=1):
        onda += amp_p * np.sin(2 * np.pi * freq * k * t)

    attack  = min(int(sr * 0.01), n // 4)
    decay   = min(int(sr * 0.05), n // 4)
    release = min(int(sr * 0.08), n // 3)
    sl      = 0.75

    env = np.ones(n) * sl
    if attack  > 0: env[:attack]               = np.linspace(0,  1,  attack)
    if decay   > 0: env[attack:attack + decay]  = np.linspace(1,  sl, decay)
    if release > 0 and release < n:
        env[-release:] = np.linspace(sl, 0, release)

    return (amplitud * onda * env).astype(np.float32)


def _nota_a_freq(nota_str: str):
    import librosa
    if nota_str in ("REST", ""):
        return None
    try:
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
            else:
                errores_nota += 1
                continue

            if not isinstance(nota_str, str):
                errores_nota += 1
                continue

            duracion = max(0.05, float(duracion))

            if nota_str == "REST":
                audio_pista.append(np.zeros(int(sr * duracion), dtype=np.float32))
                continue

            freq = _nota_a_freq(nota_str)
            if freq is None:
                errores_nota += 1
                continue

            audio_pista.append(_sintetizar_nota(freq, duracion, sr, amplitud_pista, perfil))
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
            temperatura = _temperatura_por_fase(f / num_fases, temperatura_maxima)
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