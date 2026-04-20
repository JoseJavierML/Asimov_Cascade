import random
import numpy as np
import soundfile as sf
import music21 as m21
import librosa
from memoria_base import extraer_adn_musical

def construir_cadena_markov(secuencia):
    transiciones = {}
    for i in range(len(secuencia) - 1):
        actual, siguiente = secuencia[i], secuencia[i+1]
        if actual not in transiciones: transiciones[actual] = []
        transiciones[actual].append(siguiente)
    return transiciones

def generar_nueva_obra(transiciones, longitud_base, prob_error):
    variacion = int(longitud_base * 0.1)
    longitud_final = random.randint(longitud_base - variacion, longitud_base + variacion)
    
    if not transiciones: return []
    
    notas_conocidas = list(transiciones.keys())
    estado_actual = random.choice(notas_conocidas)
    nueva_secuencia = [estado_actual]
    
    for _ in range(longitud_final - 1):
        if random.random() < prob_error:
            estado_actual = random.choice(notas_conocidas)
        elif estado_actual in transiciones and transiciones[estado_actual]:
            estado_actual = random.choice(transiciones[estado_actual])
        else:
            estado_actual = random.choice(notas_conocidas)
        nueva_secuencia.append(estado_actual)
            
    return nueva_secuencia

def sintetizar_audio_directo(diccionario_pistas, nombre_salida, sr=44100):
    audio_completo = []
    duracion_nota = 0.4 
    t = np.linspace(0, duracion_nota, int(sr * duracion_nota), False)
    max_len = 0
    pistas_audio = []
    
    for datos in diccionario_pistas.values():
        audio_pista = []
        for elem in datos['secuencia']:
            try:
                if "Acorde_" in elem:
                    n = elem.split("_")[1].split(".")[0]
                    freq = librosa.note_to_hz(m21.pitch.Pitch(int(n)).nameWithOctave)
                else:
                    freq = librosa.note_to_hz(elem)
                
                onda = 0.3 * np.sin(2 * np.pi * freq * t)
                fade = int(sr * 0.05)
                onda[:fade] *= np.linspace(0, 1, fade)
                onda[-fade:] *= np.linspace(1, 0, fade)
                audio_pista.extend(onda)
            except: continue
        pistas_audio.append(np.array(audio_pista))
        max_len = max(max_len, len(audio_pista))

    mezcla = np.zeros(max_len)
    for p in pistas_audio:
        mezcla[:len(p)] += p
    
    if np.max(np.abs(mezcla)) > 0:
        mezcla = mezcla / np.max(np.abs(mezcla))
        
    sf.write(nombre_salida, mezcla, sr)

if __name__ == "__main__":
    archivo_input = "mi_obra.wav" 
    
    if not os.path.exists(archivo_input):
        print(f"Error: No se encuentra el archivo {archivo_input}")
    else:
        memoria_actual = extraer_adn_musical(archivo_input)
        num_fases = 15
        
        print("\n=== INICIANDO CASCADA DE ASIMOV: SÍNTESIS DIRECTA ===")
        
        for f in range(1, num_fases + 1):
            p_error = (f / num_fases) * 0.35 
            print(f"Generando Fase {f:02d} (Entropía: {p_error:.1%})...")
            
            nueva_memoria = {}
            for pista, datos in memoria_actual.items():
                modelo = construir_cadena_markov(datos['secuencia'])
                nueva_sec = generar_nueva_obra(modelo, len(datos['secuencia']), p_error)
                nueva_memoria[pista] = {
                    'instrumento': datos['instrumento'],
                    'secuencia': nueva_sec
                }
            
            sintetizar_audio_directo(nueva_memoria, f"Fase_{f:02d}_Audio.wav")
            
            memoria_actual = nueva_memoria
            
        print("\n=== PROCESO COMPLETADO: Revisa tus archivos .wav ===")