import music21 as m21
import warnings
warnings.filterwarnings("ignore")

def extraer_adn_musical(archivo_midi):
    print(f"Leyendo pistas de: {archivo_midi}...\n")
    try:
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
                nombre_pista = f"Pista_{i}"
                pistas_memoria[nombre_pista] = {
                    'instrumento': instrumento,
                    'secuencia': secuencia
                }
                nombre_inst = instrumento.instrumentName or "Desconocido"
                print(f" -> Extraída {nombre_pista}: Instrumento {nombre_inst} ({len(secuencia)} eventos)")
                
        return pistas_memoria
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return {}

if __name__ == "__main__":
    archivo = "mi_obra.mid" 
    memoria = extraer_adn_musical(archivo)
    print("\n¡Lectura multipista completada!")