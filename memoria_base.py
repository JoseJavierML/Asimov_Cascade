import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def extraer_adn_desde_audio(ruta_audio: str) -> list[tuple[str, float]]:
    import librosa

    HOP = 512
    SR  = 22050
    UMBRAL_CONFIANZA = 0.15  
    MIN_DURACION     = 0.08   

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

    secuencia: list[tuple[str, float]] = []
    nota_actual = None
    dur_acumulada = 0.0
    conf_acumulada = []

    def _confirmar_nota(nota, dur, confs):
        if nota is None or dur < MIN_DURACION:
            return
        conf_media = float(np.mean(confs)) if confs else 0.0
        if conf_media < UMBRAL_CONFIANZA:
            return
        secuencia.append((nota, round(dur, 3)))

    for freq, prob in zip(f0, voiced_probs):
        if freq is None or np.isnan(freq) or prob < UMBRAL_CONFIANZA:
            _confirmar_nota(nota_actual, dur_acumulada, conf_acumulada)
            nota_actual = None
            dur_acumulada = 0.0
            conf_acumulada = []
            continue

        nota_frame = librosa.hz_to_note(float(freq))

        if nota_frame == nota_actual:
            dur_acumulada += seg_dur
            conf_acumulada.append(prob)
        else:
            _confirmar_nota(nota_actual, dur_acumulada, conf_acumulada)
            nota_actual    = nota_frame
            dur_acumulada  = seg_dur
            conf_acumulada = [prob]

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