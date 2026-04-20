import random
import music21 as m21
from memoria_base import extraer_adn_musical

def construir_cadena_markov(secuencia):
    transiciones = {}
    
    for i in range(len(secuencia) - 1):
        estado_actual = secuencia[i]
        siguiente_estado = secuencia[i+1]
        
        if estado_actual not in transiciones:
            transiciones[estado_actual] = []
        
        transiciones[estado_actual].append(siguiente_estado)
        
    return transiciones

def generar_nueva_obra(transiciones, longitud=50):
    estado_actual = random.choice(list(transiciones.keys()))
    nueva_secuencia = [estado_actual]
    
    for _ in range(longitud - 1):
        if estado_actual in transiciones and transiciones[estado_actual]:
            siguiente_estado = random.choice(transiciones[estado_actual])
            nueva_secuencia.append(siguiente_estado)
            estado_actual = siguiente_estado
        else:
            estado_actual = random.choice(list(transiciones.keys()))
            
    return nueva_secuencia

def secuencia_a_midi(secuencia, nombre_archivo_salida):
    pista = m21.stream.Stream()
    
    for elemento in secuencia:
        if elemento.startswith("Acorde_"):
            notas = elemento.split("_")[1].split(".")
            notas_reales = [m21.pitch.Pitch(int(n)).nameWithOctave for n in notas]
            acorde = m21.chord.Chord(notas_reales)
            acorde.quarterLength = 0.5 
            pista.append(acorde)
        else:
            nota = m21.note.Note(elemento)
            nota.quarterLength = 0.5
            pista.append(nota)
    pista.write('midi', fp=nombre_archivo_salida)

if __name__ == "__main__":
    archivo_original = "mi_obra.mid"
    
    memoria = extraer_adn_musical(archivo_original)
    
    if memoria:
        print("\nConstruyendo la red neuronal (Cadena de Markov)...")
        modelo = construir_cadena_markov(memoria)
        
        print("Generando la Fase 1 del deterioro (50 eventos musicales)...")
        nueva_cancion = generar_nueva_obra(modelo, longitud=50)
        
        nombre_salida = "Fase_1_Recuerdo.mid"
        secuencia_a_midi(nueva_cancion, nombre_salida)
        
        print(f"¡Éxito! Se ha guardado el archivo: {nombre_salida}")
        print("Abre el archivo en tu reproductor o programa de música y escucha cómo te recuerda la máquina.")