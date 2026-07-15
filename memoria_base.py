import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def extraer_adn_desde_audio(ruta_audio: str) -> list[tuple[str, float, float]]:
    import librosa
    from scipy.signal import find_peaks

    HOP = 512
    SR  = 22050
    UMBRAL_CONFIANZA = 0.15
    MIN_DURACION     = 0.05
    UMBRAL_SILENCIO_DB = -45.0
    TOL_ONSET_FRAMES   = 1
    UMBRAL_ONSET_DELTA = 0.08

    print(f"  [Audio] Cargando: {os.path.basename(ruta_audio)}")
    y, sr = librosa.load(ruta_audio, sr=SR)
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
        pre_avg=4,
        post_avg=4,
        delta=UMBRAL_ONSET_DELTA,
        wait=3,
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

    seg_dur = HOP / SR  
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

    secuencia: list[tuple[str, float, float]] = []

    print(f"  [Audio] Tempo global estimado: {tempo_estimado:.2f} BPM")

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
            return
        energia_segmento = np.mean(cqt[:, inicio:fin], axis=1)
        if not np.any(np.isfinite(energia_segmento)) or float(np.max(energia_segmento)) <= 0:
            return
        prominencia_minima = max(float(np.max(energia_segmento)) * 0.03, 1e-8)
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

    print(f"  [Audio] Extraídos {len(secuencia)} eventos (duración mínima {MIN_DURACION*1000:.0f} ms).")
    if len(secuencia) < 10:
        print("  [Audio] AVISO: muy pocas notas detectadas. Prueba con un archivo MIDI para mejores resultados.")
    return secuencia


def extraer_adn_desde_midi(archivo_midi: str) -> dict:
    import music21 as m21

    print(f"  [MIDI] Leyendo: {os.path.basename(archivo_midi)}")
    partitura = m21.converter.parse(archivo_midi)
    pistas_memoria: dict = {}

    for i, part in enumerate(partitura.parts):
        instrumento = part.getInstrument()
        secuencia: list[tuple[str, float]] = []
        prev_offset = 0.0

        for elemento in part.chordify().flatten().notesAndRests:
            offset_actual = float(elemento.offset)
            duracion_q = float(elemento.duration.quarterLength)

            gap = offset_actual - prev_offset
            if gap > 0.05:
                secuencia.append(("REST", round(gap, 3)))

            if isinstance(elemento, m21.note.Rest):
                secuencia.append(("REST", round(duracion_q, 3)))
            elif isinstance(elemento, m21.chord.Chord):
                notas_acorde = ".".join(
                    str(n.pitch.midi) for n in sorted(elemento.notes, key=lambda x: x.pitch.midi)
                )
                secuencia.append((f"Acorde_{notas_acorde}", round(duracion_q, 3)))
            elif isinstance(elemento, m21.note.Note):
                secuencia.append((str(elemento.pitch), round(duracion_q, 3)))

            prev_offset = offset_actual + duracion_q

        if secuencia:
            nombre_inst = instrumento.instrumentName or "Piano"
            pistas_memoria[f"Pista_{i}_{nombre_inst}"] = {
                "instrumento": instrumento,
                "secuencia": secuencia,
            }
            print(f"  [MIDI] Pista {i} ({nombre_inst}): {len(secuencia)} eventos.")

    return pistas_memoria


def extraer_adn_musical(archivo: str) -> dict:
    import music21 as m21

    ext = os.path.splitext(archivo)[1].lower()

    if ext in (".wav", ".mp3"):
        secuencia = extraer_adn_desde_audio(archivo)
        return {
            "Audio_Principal": {
                "instrumento": m21.instrument.Piano(),
                "secuencia": secuencia,
            }
        }

    elif ext in (".mid", ".midi"):
        return extraer_adn_desde_midi(archivo)

    else:
        raise ValueError(f"Formato no soportado: '{ext}'. Usa .mid, .midi, .wav o .mp3")