import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def extraer_adn_desde_audio(ruta_audio: str) -> list[tuple[str, float]]:
    import librosa
    from scipy.ndimage import median_filter

    HOP = 512
    SR  = 22050
    UMBRAL_CONFIANZA = 0.15  
    MIN_DURACION     = 0.08   
    VENTANA_MEDIANA  = 7
    MAX_GAP_FRAMES   = 2

    print(f"  [Audio] Cargando: {os.path.basename(ruta_audio)}")
    y, sr = librosa.load(ruta_audio, sr=SR)

    f0, _voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=2048,
        hop_length=HOP,
        fill_na=None,
    )

    seg_dur = HOP / SR  

    midi_crudo = np.full(len(f0), np.nan, dtype=float)
    for idx, (freq, prob) in enumerate(zip(f0, voiced_probs)):
        if freq is None or np.isnan(freq) or prob < UMBRAL_CONFIANZA:
            continue
        midi_crudo[idx] = float(librosa.hz_to_midi(float(freq)))

    midi_suavizado = np.full_like(midi_crudo, np.nan)
    indices_voz = np.flatnonzero(~np.isnan(midi_crudo))
    if indices_voz.size:
        cortes = np.where(np.diff(indices_voz) != 1)[0] + 1
        for segmento in np.split(indices_voz, cortes):
            valores = midi_crudo[segmento]
            if len(valores) >= 3:
                ventana = min(VENTANA_MEDIANA, len(valores))
                if ventana % 2 == 0:
                    ventana -= 1
                if ventana < 3:
                    ventana = 3
                valores = median_filter(valores, size=ventana, mode="nearest")
            midi_suavizado[segmento] = valores

    notas_frames: list[str | None] = []
    for midi_val in midi_suavizado:
        if np.isnan(midi_val):
            notas_frames.append(None)
            continue
        midi_entero = int(np.rint(midi_val))
        notas_frames.append(librosa.midi_to_note(midi_entero, octave=True, cents=False))

    if notas_frames:
        i = 0
        while i < len(notas_frames):
            if notas_frames[i] is not None:
                i += 1
                continue
            inicio = i
            while i < len(notas_frames) and notas_frames[i] is None:
                i += 1
            fin = i
            hueco = fin - inicio
            nota_izq = notas_frames[inicio - 1] if inicio > 0 else None
            nota_der = notas_frames[fin] if fin < len(notas_frames) else None
            if nota_izq is not None and nota_izq == nota_der and hueco <= MAX_GAP_FRAMES:
                for j in range(inicio, fin):
                    notas_frames[j] = nota_izq

    secuencia: list[tuple[str, float]] = []
    nota_actual = None
    dur_acumulada = 0.0
    conf_acumulada = []
    hueco_acumulado = 0.0
    huecos_consecutivos = 0

    def _confirmar_nota(nota, dur, confs):
        if nota is None or dur < MIN_DURACION:
            return
        conf_media = float(np.mean(confs)) if confs else 0.0
        if conf_media < UMBRAL_CONFIANZA:
            return
        secuencia.append((nota, round(dur, 3)))

    for nota_frame, prob in zip(notas_frames, voiced_probs):
        if nota_frame is None:
            if nota_actual is not None:
                hueco_acumulado += seg_dur
                huecos_consecutivos += 1
                if huecos_consecutivos > MAX_GAP_FRAMES:
                    _confirmar_nota(nota_actual, dur_acumulada, conf_acumulada)
                    nota_actual = None
                    dur_acumulada = 0.0
                    conf_acumulada = []
                    hueco_acumulado = 0.0
                    huecos_consecutivos = 0
            continue

        if nota_actual is None:
            nota_actual = nota_frame
            dur_acumulada = seg_dur
            conf_acumulada = [prob]
            hueco_acumulado = 0.0
            huecos_consecutivos = 0
            continue

        if nota_frame == nota_actual:
            dur_acumulada += seg_dur + hueco_acumulado
            conf_acumulada.append(prob)
        else:
            _confirmar_nota(nota_actual, dur_acumulada, conf_acumulada)
            nota_actual = nota_frame
            dur_acumulada = seg_dur
            conf_acumulada = [prob]

        hueco_acumulado = 0.0
        huecos_consecutivos = 0

    _confirmar_nota(nota_actual, dur_acumulada, conf_acumulada)

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