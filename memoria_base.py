import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def _aislar_stems_demucs(ruta_audio: str):
    import importlib.util
    import pathlib
    import subprocess
    import sys
    import tempfile
    import traceback

    if importlib.util.find_spec("demucs") is None:
        print("  [Audio] AVISO: demucs no está instalado; se omite la separación de stems.")
        return {"master": ruta_audio}, None

    temp_dir = tempfile.TemporaryDirectory(prefix="asimov_demucs_")
    comando = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs_6s",
        "--device",
        "cpu",
        "--out",
        temp_dir.name,
        ruta_audio,
    ]

    try:
        resultado = subprocess.run(
            comando,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if resultado.stdout:
            print("  [Audio] Demucs:", resultado.stdout.splitlines()[-1])
    except Exception as exc:
        motivo = "error desconocido"
        stderr_texto = ""
        stdout_texto = ""

        if isinstance(exc, subprocess.TimeoutExpired):
            motivo = "timeout durante la separación"
            stderr_texto = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            stdout_texto = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        elif isinstance(exc, subprocess.CalledProcessError):
            stderr_texto = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            stdout_texto = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr_lower = stderr_texto.lower()
            if "ffmpeg" in stderr_lower and ("not found" in stderr_lower or "no such file" in stderr_lower):
                motivo = "FFMPEG no disponible"
            elif "out of memory" in stderr_lower or "cuda out of memory" in stderr_lower:
                motivo = "memoria insuficiente (OOM)"
            else:
                motivo = f"demucs terminó con código {exc.returncode}"
        else:
            motivo = f"{type(exc).__name__}: {exc}"

        print("  [Audio] ===== DIAGNÓSTICO DE FALLO DEMUCS =====")
        print(f"  [Audio] Comando ejecutado: {' '.join(comando)}")
        print(f"  [Audio] Motivo detectado: {motivo}")
        if stdout_texto.strip():
            print("  [Audio] STDOUT Demucs:")
            print(stdout_texto.strip())
        if stderr_texto.strip():
            print("  [Audio] STDERR Demucs:")
            print(stderr_texto.strip())
        print("  [Audio] Traceback completo:")
        traceback.print_exc()
        print("  [Audio] !!! ADVERTENCIA CRÍTICA: Demucs abortó; se aplicará fallback a pista master. !!!")
        temp_dir.cleanup()
        return {"master": ruta_audio}, None

    stems_retenidos: dict[str, str] = {}
    for nombre_stem in ("other", "bass", "drums", "piano", "guitar"):
        candidatos = sorted(pathlib.Path(temp_dir.name).rglob(f"{nombre_stem}.wav"))
        if candidatos:
            stems_retenidos[nombre_stem] = str(candidatos[0])

    if not stems_retenidos:
        print("  [Audio] AVISO: demucs no produjo stems retenidos. Se usará el audio original.")
        temp_dir.cleanup()
        return {"master": ruta_audio}, None

    return stems_retenidos, temp_dir


def _instrumento_desde_stem(nombre_stem: str):
    import music21 as m21

    instrumento = m21.instrument.Instrument()
    if nombre_stem == "piano":
        instrumento.instrumentName = "Piano"
    elif nombre_stem == "guitar":
        instrumento.instrumentName = "Guitar"
    elif nombre_stem == "other":
        instrumento.instrumentName = "Pad"
    elif nombre_stem == "bass":
        instrumento.instrumentName = "Bass"
    elif nombre_stem == "drums":
        instrumento.instrumentName = "Drums"
    else:
        instrumento.instrumentName = nombre_stem.title()
    return instrumento


def _log_resumen_eventos_stems(pistas_memoria: dict):
    for stem, datos_stem in pistas_memoria.items():
        eventos = datos_stem.get("secuencia", [])
        total_notas_acordes = sum(1 for nota, _dur, _vel in eventos if nota != "REST")
        print(
            f"  [Audio] Stem '{stem}': {len(eventos)} eventos extraídos "
            f"({total_notas_acordes} notas/acordes)."
        )


def _tempo_del_mix(ruta_audio: str) -> float:
    """Tempo global medido una sola vez sobre la mezcla completa.

    Antes cada stem pasaba por beat_track por su cuenta. En una pieza real eso
    devuelve valores distintos para el piano, el bajo y la bateria, de modo que
    las pistas ya entran desincronizadas en la Fase 1 y el 'desincronizar los
    cerebros' que describe el README ocurre por error, no por temperatura.
    """
    import librosa

    try:
        y, sr = librosa.load(ruta_audio, sr=22050, duration=None)
        if y.size == 0:
            return 120.0
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        tempo = float(np.asarray(tempo).reshape(-1)[0])
        if np.isfinite(tempo) and tempo > 0:
            return round(tempo, 3)
    except Exception:
        pass
    return 120.0


def _extraer_superestado_desde_audio(ruta_audio: str, nombre_stem: str, bpm_global: float = None):
    import librosa
    from scipy.signal import find_peaks

    HOP = 512
    SR = 22050
    MIN_DURACION = 0.05
    UMBRAL_SILENCIO_DB = -55.0
    TOL_ONSET_FRAMES = 1
    UMBRAL_ONSET_DELTA = 0.03

    print(f"  [Audio] {nombre_stem}: cargando {os.path.basename(ruta_audio)}")
    y_original, sr = librosa.load(ruta_audio, sr=SR)
    y, trim_idx = librosa.effects.trim(y_original, top_db=50)
    if y.size == 0:
        y = y_original
        trim_idx = np.array([0, len(y_original)], dtype=int)

    frames_recortados = int(trim_idx[0])
    if frames_recortados > 0:
        segundos_recortados = frames_recortados / sr
        print(f"  [Audio] {nombre_stem}: recorte inicial de silencio {segundos_recortados:.3f}s")

    if bpm_global and bpm_global > 0:
        tempo_estimado = float(bpm_global)
    else:
        tempo_estimado, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
        tempo_estimado = float(np.asarray(tempo_estimado).reshape(-1)[0])
        if not np.isfinite(tempo_estimado) or tempo_estimado <= 0:
            tempo_estimado = 120.0

    y_harmonic, _ = librosa.effects.hpss(y)

    frame_length = 2048
    rms = librosa.feature.rms(y=y_harmonic, frame_length=frame_length, hop_length=HOP, center=True)[0]
    db_frames = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=np.max)

    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=HOP,
        units="frames",
        backtrack=False,
        pre_max=1,
        post_max=1,
        pre_avg=2,
        post_avg=2,
        delta=UMBRAL_ONSET_DELTA,
        wait=1,
    )

    cqt = np.abs(
        librosa.cqt(
            y_harmonic,
            sr=SR,
            hop_length=HOP,
            fmin=librosa.note_to_hz("C2"),
            n_bins=5 * 12,
            bins_per_octave=12,
        )
    )
    cqt_freqs = librosa.cqt_frequencies(5 * 12, fmin=librosa.note_to_hz("C2"), bins_per_octave=12)

    n_frames = min(cqt.shape[1], len(db_frames))
    cqt = cqt[:, :n_frames]
    db_frames = db_frames[:n_frames]
    rms = rms[:n_frames]

    if len(onsets):
        onsets = np.unique(onsets[onsets < n_frames])

    silencio_frames = db_frames <= UMBRAL_SILENCIO_DB

    seg_dur = HOP / sr
    negra_segundos = 60.0 / tempo_estimado
    subduraciones_base = np.array([4.0, 2.0, 1.0, 0.5, 0.25], dtype=float) * negra_segundos
    rejilla_subdivision = np.unique(
        np.concatenate(
            [
                subduraciones_base,
                subduraciones_base * 1.5,
                subduraciones_base * (2.0 / 3.0),
            ]
        )
    )

    print(f"  [Audio] {nombre_stem}: tempo global estimado {tempo_estimado:.2f} BPM")

    def _cuantizar_duracion(duracion: float) -> float:
        if duracion <= 0:
            return 0.0

        return float(rejilla_subdivision[int(np.argmin(np.abs(rejilla_subdivision - duracion)))])

    def _normalizar_velocidades(eventos_crudos: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
        energias = [energia for nota, _dur, energia in eventos_crudos if nota != "REST"]
        if not energias:
            return [(nota, dur, 0.0 if nota == "REST" else 1.0) for nota, dur, _energia in eventos_crudos]

        energia_min = float(np.min(energias))
        energia_max = float(np.max(energias))

        def _velocidad_desde_energia(energia: float) -> float:
            if energia_max <= energia_min + 1e-12:
                return 1.0
            return float(np.clip((energia - energia_min) / (energia_max - energia_min), 0.0, 1.0))

        eventos_normalizados: list[tuple[str, float, float]] = []
        for nota, dur, energia in eventos_crudos:
            if nota == "REST":
                eventos_normalizados.append((nota, dur, 0.0))
            else:
                eventos_normalizados.append((nota, dur, _velocidad_desde_energia(energia)))
        return eventos_normalizados

    def _agregar_rest(secuencia_local: list[tuple[str, float, float]], duracion: float):
        duracion_cuantizada = _cuantizar_duracion(duracion)
        if duracion_cuantizada > 0:
            secuencia_local.append(("REST", round(duracion_cuantizada, 6), 0.0))

    def _notas_desde_segmento(inicio: int, fin: int) -> tuple[str, ...] | None:
        if fin <= inicio:
            return None
        energia_segmento = np.mean(cqt[:, inicio:fin], axis=1)
        if not np.any(np.isfinite(energia_segmento)) or float(np.max(energia_segmento)) <= 0:
            return None
        prominencia_minima = max(float(np.max(energia_segmento)) * 0.015, 1e-8)
        picos, propiedades = find_peaks(energia_segmento, prominence=prominencia_minima, distance=1)

        candidatos: list[int] = []
        if picos.size:
            orden = np.argsort(propiedades["prominences"])[::-1]
            candidatos.extend(int(picos[i]) for i in orden)

        if len(candidatos) < 4:
            orden_energia = np.argsort(energia_segmento)[::-1]
            candidatos.extend(int(indice) for indice in orden_energia)

        notas: list[str] = []
        midi_vistos: set[int] = set()
        for bin_idx in candidatos:
            if len(notas) >= 4:
                break
            if bin_idx < 0 or bin_idx >= len(cqt_freqs):
                continue

            freq = float(cqt_freqs[bin_idx])
            if freq <= 0:
                continue

            midi_val = int(np.rint(librosa.hz_to_midi(freq)))
            midi_val = int(np.clip(midi_val, librosa.note_to_midi("C2"), librosa.note_to_midi("C6")))
            if midi_val in midi_vistos:
                continue

            notas.append(librosa.midi_to_note(midi_val, octave=True, cents=False))
            midi_vistos.add(midi_val)

        if not notas:
            return None

        return tuple(sorted(notas, key=lambda n: librosa.note_to_midi(n)))

    def _extraer_con_onsets():
        secuencia_local: list[tuple[str, float, float]] = []
        ultimo_fin = 0

        for indice, onset in enumerate(onsets):
            inicio = max(0, int(onset) - TOL_ONSET_FRAMES)
            if inicio >= n_frames:
                continue

            siguiente_onset = int(onsets[indice + 1]) if indice + 1 < len(onsets) else n_frames
            limite = min(n_frames, max(inicio + 1, siguiente_onset))
            silencios = np.flatnonzero(silencio_frames[inicio:limite])
            fin = inicio + int(silencios[0]) if silencios.size else limite

            if fin - inicio < 1:
                continue

            nota = _notas_desde_segmento(inicio, fin)
            duracion = (fin - inicio) * seg_dur
            if nota is None or duracion < MIN_DURACION:
                continue

            energia_inicio = max(0, inicio - 1)
            energia_fin = min(n_frames, inicio + 3)
            energia_segmento = float(np.mean(rms[energia_inicio:energia_fin])) if energia_fin > energia_inicio else 0.0

            if inicio > ultimo_fin:
                _agregar_rest(secuencia_local, (inicio - ultimo_fin) * seg_dur)

            duracion_cuantizada = _cuantizar_duracion(duracion)
            secuencia_local.append((nota, round(duracion_cuantizada, 6), energia_segmento))
            ultimo_fin = fin

        return _normalizar_velocidades(secuencia_local)

    def _extraer_sin_onsets():
        secuencia_local: list[tuple[str, float, float]] = []
        ultimo_fin = 0
        notas_validas = ~silencio_frames

        if not np.any(notas_validas):
            return secuencia_local

        indices_voz = np.flatnonzero(notas_validas)
        cortes = np.where(np.diff(indices_voz) != 1)[0] + 1

        for segmento in np.split(indices_voz, cortes):
            if segmento.size == 0:
                continue

            inicio = int(segmento[0])
            fin = int(segmento[-1]) + 1
            nota = _notas_desde_segmento(inicio, fin)
            duracion = (fin - inicio) * seg_dur
            if nota is None or duracion < MIN_DURACION:
                continue

            energia_segmento = float(np.mean(rms[inicio:fin])) if fin > inicio else 0.0

            if inicio > ultimo_fin:
                _agregar_rest(secuencia_local, (inicio - ultimo_fin) * seg_dur)

            duracion_cuantizada = _cuantizar_duracion(duracion)
            secuencia_local.append((nota, round(duracion_cuantizada, 6), energia_segmento))
            ultimo_fin = fin

        return _normalizar_velocidades(secuencia_local)

    secuencia = _extraer_con_onsets()
    if not secuencia:
        secuencia = _extraer_sin_onsets()

    print(f"  [Audio] {nombre_stem}: extraídos {len(secuencia)} eventos (duración mínima {MIN_DURACION*1000:.0f} ms).")
    if len(secuencia) < 10:
        print("  [Audio] AVISO: muy pocas notas detectadas. Prueba con un archivo MIDI para mejores resultados.")
    return secuencia, round(float(tempo_estimado), 3)


def extraer_adn_desde_audio(ruta_audio: str) -> dict:
    stems, separacion_tmp = _aislar_stems_demucs(ruta_audio)

    try:
        if "master" in stems:
            secuencia_master, bpm_master = _extraer_superestado_desde_audio(stems["master"], "master")
            resultado_master = {
                "master": {
                    "instrumento": _instrumento_desde_stem("master"),
                    "secuencia": secuencia_master,
                    "bpm": bpm_master,
                }
            }
            _log_resumen_eventos_stems(resultado_master)
            return resultado_master

        pistas_memoria: dict = {}
        bpm_global = _tempo_del_mix(ruta_audio)
        print(f"  [Audio] Tempo global de la mezcla: {bpm_global:.2f} BPM")

        for nombre_stem in ("piano", "guitar", "other", "bass", "drums"):
            ruta_stem = stems.get(nombre_stem)
            if not ruta_stem:
                continue

            secuencia, _bpm_stem = _extraer_superestado_desde_audio(ruta_stem, nombre_stem, bpm_global)
            if not secuencia:
                continue

            pistas_memoria[nombre_stem] = {
                "instrumento": _instrumento_desde_stem(nombre_stem),
                "secuencia": secuencia,
                "bpm": bpm_global,
            }

        if not pistas_memoria:
            print("  [Audio] AVISO: ningún stem produjo eventos útiles. Se usará el audio original como master.")
            secuencia_master, bpm_master = _extraer_superestado_desde_audio(ruta_audio, "master")
            resultado_master = {
                "master": {
                    "instrumento": _instrumento_desde_stem("master"),
                    "secuencia": secuencia_master,
                    "bpm": bpm_master,
                }
            }
            _log_resumen_eventos_stems(resultado_master)
            return resultado_master

        _log_resumen_eventos_stems(pistas_memoria)
        return pistas_memoria
    finally:
        if separacion_tmp is not None:
            separacion_tmp.cleanup()


def _velocidad_midi(elemento) -> float:
    """Dinamica real de una nota MIDI, normalizada a 0..1.

    Antes se fijaba siempre en 1.0, lo que hacia que el tercer campo del
    Super-Estado (Velocidad_RMS) fuera ficticio en toda la ruta MIDI. Se
    normaliza como en la ruta de audio, donde ya venia en 0..1.
    """
    import music21 as m21

    if isinstance(elemento, m21.note.Rest):
        return 0.0

    try:
        valor = elemento.volume.velocity
        if valor is None:
            return 1.0
        midi = float(getattr(valor, "velocity", valor))
    except Exception:
        return 1.0

    if not (0.0 <= midi <= 127.0):
        return 1.0
    return round(float(min(1.0, max(0.05, midi / 127.0))), 4)


_PALABRAS_MODO = (
    ("drum", "drums"), ("perc", "drums"), ("kit", "drums"),
    ("bass", "bass"), ("sub", "bass"),
    ("guitar", "guitar"), ("pluck", "guitar"),
    ("violin", "strings"), ("cello", "strings"), ("string", "strings"),
    ("organ", "organ"), ("harm", "organ"),
    ("pad", "other"), ("bell", "other"), ("synth", "other"),
)


def _modo_de_pista(nombre: str, secuencia: list) -> str:
    """Modo de sintesis de una pista MIDI: manda el nombre, si no, el registro.

    Al releer un MIDI, music21 deja instrument.instrumentName a None y el nombre
    sobrevive solo en part.partName. Leer unicamente el segundo renombraba TODAS
    las pistas a 'Piano', y el sintetizador de bajo no se activaba nunca.
    """
    texto = (nombre or "").strip().lower()
    for palabra, modo in _PALABRAS_MODO:
        if palabra in texto:
            return modo
    return _modo_por_registro(secuencia)


def _modo_por_registro(secuencia: list, umbral_bajo: float = 53.0) -> str:
    """Infiere el papel de la pista por su registro medio.

    Muchos MIDI reales llegan sin nombre de instrumento util, y entonces no hay
    forma de distinguir un bajo de un piano por otravia que el registro que
    ocupa. Un Do2 medio solo puede ser un bajo.
    """
    from motor_generativo import _a_midi

    midis = [midi for nota, _dur, _vel in secuencia for midi in _a_midi(nota)]
    if not midis:
        return "piano"

    mediana = float(np.median(midis))
    if mediana < umbral_bajo:
        return "bass"
    if mediana > 72:
        return "other"
    return "piano"


def _mapa_tempo_musical(partitura) -> list:
    """Extrae el mapa de tempo del MIDI como [(offset_cuadradas, segundos_por_cuadrada)].

    music21 guarda el tempo como marcas (MetronomeMark) que pueden cambiar a
    mitad de la obra. Se ordenan e incluyen el origen en la posicion 0.
    Sin marcas se asume 120 BPM, que es la convencion MIDI estandar.
    """
    import music21 as m21

    marcas: list = []
    try:
        for marca in partitura.flatten().getElementsByClass(m21.tempo.MetronomeMark):
            bpm = float(marca.number) if marca.number else 0.0
            if bpm <= 0:
                continue
            referencia = marca.referent
            ref_cuadradas = float(referencia.quarterLength) if referencia is not None else 1.0
            if ref_cuadradas <= 0:
                ref_cuadradas = 1.0
            marcas.append((float(marca.offset), 60.0 * ref_cuadradas / bpm))
    except Exception:
        marcas = []

    marcas = sorted({(round(offset, 6), round(segundos, 9)) for offset, segundos in marcas})

    if not marcas or marcas[0][0] > 0:
        marcas.insert(0, (0.0, 0.5))  # 120 BPM hasta el primer cambio

    return marcas


def _segundos_por_cuadrada(mapa: list, offset: float) -> float:
    valor = mapa[0][1]
    for posicion, segundos in mapa:
        if posicion <= offset:
            valor = segundos
        else:
            break
    return valor


def _duracion_en_segundos(mapa: list, inicio: float, duracion_cuadradas: float) -> float:
    """Convierte duracion en negras a segundos, integrando cambios de tempo.

    Una nota que cruza un cambio de tempo se divide en tramos y cada tramo usa
    el tempo que estava vigente. Sin esto, un MIDI con accelerando se
    estiraria o se comprimiria de forma incorrecta.
    """
    if duracion_cuadradas <= 0:
        return 0.0

    fin = inicio + duracion_cuadradas
    cortes = [inicio] + [pos for pos, _ in mapa if inicio < pos < fin] + [fin]

    total = 0.0
    for tramo_a, tramo_b in zip(cortes[:-1], cortes[1:]):
        if tramo_b <= tramo_a:
            continue
        total += (tramo_b - tramo_a) * _segundos_por_cuadrada(mapa, tramo_a)
    return total


def extraer_adn_desde_midi(archivo_midi: str) -> dict:
    import music21 as m21

    print(f"  [MIDI] Leyendo: {os.path.basename(archivo_midi)}")
    partitura = m21.converter.parse(archivo_midi)
    mapa_tempo = _mapa_tempo_musical(partitura)
    segundos_por_cuadrada = mapa_tempo[0][1]
    bpm_principal = 60.0 / segundos_por_cuadrada if segundos_por_cuadrada > 0 else 120.0

    if len(mapa_tempo) > 1:
        print(f"  [MIDI] Mapa de tempo: {len(mapa_tempo)} marcas (principal {bpm_principal:.1f} BPM)")
    else:
        print(f"  [MIDI] Tempo detectado: {bpm_principal:.1f} BPM "
              f"({segundos_por_cuadrada:.4f} s por negra)")

    pistas_memoria: dict = {}

    for i, part in enumerate(partitura.parts):
        instrumento = part.getInstrument()
        secuencia: list = []
        prev_offset = 0.0

        for elemento in part.chordify().flatten().notesAndRests:
            offset_actual = float(elemento.offset)
            duracion_q = float(elemento.duration.quarterLength)

            gap = offset_actual - prev_offset
            if gap > 0.05:
                gap_seg = _duracion_en_segundos(mapa_tempo, prev_offset, gap)
                secuencia.append(("REST", round(gap_seg, 4), 0.0))

            duracion_seg = _duracion_en_segundos(mapa_tempo, offset_actual, duracion_q)
            velocidad = _velocidad_midi(elemento)

            if isinstance(elemento, m21.note.Rest):
                secuencia.append(("REST", round(duracion_seg, 4), 0.0))
            elif isinstance(elemento, m21.chord.Chord):
                notas_acorde = ".".join(
                    str(n.pitch.midi) for n in sorted(elemento.notes, key=lambda x: x.pitch.midi)
                )
                secuencia.append((f"Acorde_{notas_acorde}", round(duracion_seg, 4), velocidad))
            elif isinstance(elemento, m21.note.Note):
                secuencia.append((str(elemento.pitch), round(duracion_seg, 4), velocidad))

            prev_offset = offset_actual + duracion_q

        if secuencia:
            # Al releer un MIDI, music21 conserva el nombre en part.partName pero
            # deja instrument.instrumentName a None, asi que leer solo el segundo
            # renombraba TODAS las pistas a "Piano".
            nombre_bruto = (getattr(part, "partName", None) or "").strip()
            if not nombre_bruto:
                nombre_bruto = (getattr(instrumento, "instrumentName", None) or "").strip()

            modo = _modo_de_pista(nombre_bruto, secuencia)
            nombre_inst = nombre_bruto.title() if nombre_bruto else {
                "bass": "Bajo", "other": "Pad", "piano": "Piano"
            }.get(modo, "Piano")

            pistas_memoria[f"Pista_{i}_{nombre_inst}"] = {
                "instrumento": instrumento,
                "secuencia": secuencia,
                "bpm": bpm_principal,
                "segundos_por_cuadrada": segundos_por_cuadrada,
                "modo_sintesis": modo,
            }
            total_seg = sum(float(d) for _n, d, _v in secuencia)
            print(f"  [MIDI] Pista {i} ({nombre_inst}): {len(secuencia)} eventos "
                  f"({total_seg:.1f}s a {bpm_principal:.1f} BPM)")

    return pistas_memoria


def extraer_adn_musical(archivo: str) -> dict:
    ext = os.path.splitext(archivo)[1].lower()

    if ext in (".wav", ".mp3"):
        return extraer_adn_desde_audio(archivo)

    elif ext in (".mid", ".midi"):
        return extraer_adn_desde_midi(archivo)

    else:
        raise ValueError(f"Formato no soportado: '{ext}'. Usa .mid, .midi, .wav o .mp3")