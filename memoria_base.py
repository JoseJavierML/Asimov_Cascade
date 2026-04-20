import music21 as m21
import warnings
warnings.filterwarnings("ignore") 

def extraer_adn_musical(archivo_midi):
    print(f"Leyendo la obra original: {archivo_midi}...\n")
    
    try:
        partitura = m21.converter.parse(archivo_midi)
        notas_y_acordes = partitura.chordify().flatten().notes
        
        secuencia_musical = []
        
        for elemento in notas_y_acordes:
            if isinstance(elemento, m21.chord.Chord):
                notas_acorde = '.'.join(str(n) for n in elemento.normalOrder)
                secuencia_musical.append(f"Acorde_{notas_acorde}")
            elif isinstance(elemento, m21.note.Note):
                secuencia_musical.append(str(elemento.pitch))

        return secuencia_musical
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return []

archivo = "mi_obra.mid" 
memoria = extraer_adn_musical(archivo)

if memoria:
    print("Los primeros 20 eventos en la memoria de la máquina son:")
    print(memoria[:20])