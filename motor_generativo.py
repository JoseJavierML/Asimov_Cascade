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

def generar_nueva_obra(transiciones, longitud_base, probabilidad_error):
    variacion = int(longitud_base * 0.1)
    longitud_final = random.randint(longitud_base - variacion, longitud_base + variacion)
    
    if not transiciones: return []
    
    todas_las_notas_vistas = list(transiciones.keys())
    estado_actual = random.choice(todas_las_notas_vistas)
    nueva_secuencia = [estado_actual]
    
    for _ in range(longitud_final - 1):
        if random.random() < probabilidad_error:
            siguiente_estado = random.choice(todas_las_notas_vistas)
        elif estado_actual in transiciones and transiciones[estado_actual]:
            siguiente_estado = random.choice(transiciones[estado_actual])
        else:
            siguiente_estado = random.choice(todas_las_notas_vistas)
            
        nueva_secuencia.append(siguiente_estado)
        estado_actual = siguiente_estado
            
    return nueva_secuencia

def guardar_multipista_a_midi(diccionario_pistas, nombre_archivo_salida):
    partitura_final = m21.stream.Score()
    
    for nombre_pista, datos in diccionario_pistas.items():
        pista = m21.stream.Part()
        
        inst = datos['instrumento']
        if inst:
            pista.insert(0, inst)
            
        for elemento in datos['secuencia']:
            try:
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
            except:
                continue
                
        partitura_final.insert(0, pista)
        
    partitura_final.write('midi', fp=nombre_archivo_salida)


if __name__ == "__main__":
    archivo_original = "mi_obra.mid"
    memoria_actual_multipista = extraer_adn_musical(archivo_original)
    
    if memoria_actual_multipista:
        numero_de_fases = 15 
        print("\n=== INICIANDO LA CASCADA DE ASIMOV  ===")
        
        for fase in range(1, numero_de_fases + 1):
            prob_error = (fase / numero_de_fases) * 0.3 
            
            print(f"\nGenerando Fase {fase} (Prob. de Error: {prob_error:.2%})...")
            
            nueva_memoria_multipista = {}
            for nombre_pista, datos in memoria_actual_multipista.items():
                secuencia_original = datos['secuencia']
                modelo = construir_cadena_markov(secuencia_original)
                nueva_sec = generar_nueva_obra(modelo, len(secuencia_original), prob_error)
                
                nueva_memoria_multipista[nombre_pista] = {
                    'instrumento': datos['instrumento'],
                    'secuencia': nueva_sec
                }
            
            nombre_salida = f"Fase_{fase:02d}_Asimov.mid" 
            guardar_multipista_a_midi(nueva_memoria_multipista, nombre_salida)
            memoria_actual_multipista = nueva_memoria_multipista
            
        print("\n=== PROCESO FINALIZADO ===")