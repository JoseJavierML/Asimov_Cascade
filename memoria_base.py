import music21 as m21
import librosa
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

def extraer_adn_desde_audio(ruta_audio):
    print(f"--- Analizando señal de audio: {os.path.basename(ruta_audio)} ---")
    y, sr = librosa.load(ruta_audio)
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7')
    )
    
    secuencia = []
    for freq in f0:
        if not np.isnan(freq) and freq > 0:
            nota = librosa.hz_to_note(freq)
            if not secuencia or secuencia[-1] != nota:
                secuencia.append(nota)
    
    print(f" -> Extraídas {len(secuencia)} notas del audio.")
    return secuencia

def extraer_adn_desde_midi(archivo_midi):
    print(f"--- Leyendo pistas MIDI: {os.path.basename(archivo_midi)} ---")
    partitura = m21.converter.parse(archivo_midi)
    pistas_memoria = {}
    
    for i, part in enumerate(partitura.parts):
        instrumento = part.getInstrument()
        secuencia = []
        for elemento in part.chordify().flatten().notes:
            if isinstance(elemento, m21.chord.Chord):
                notas_acorde = '.'.join(str(n) for n in elemento.normalOrder)
                secuencia.append(f"Acorde_{notas_acorde}")
            elif isinstance(elemento, m21.note.Note):
                secuencia.append(str(elemento.pitch))
        
        if secuencia:
            pistas_memoria[f"Pista_{i}"] = {
                'instrumento': instrumento,
                'secuencia': secuencia
            }
            nombre_inst = instrumento.instrumentName or "Generic"
            print(f" -> Pista {i} ({nombre_inst}): {len(secuencia)} eventos.")
            
    return pistas_memoria

def extraer_adn_musical(archivo):
    ext = os.path.splitext(archivo)[1].lower()
    if ext in ['.wav', '.mp3']:
        secuencia = extraer_adn_desde_audio(archivo)
        return {"Audio_Principal": {"instrumento": m21.instrument.Piano(), "secuencia": secuencia}}
    elif ext in ['.mid', '.midi']:
        return extraer_adn_desde_midi(archivo)
    else:
        print("Formato no soportado.")
        return {}