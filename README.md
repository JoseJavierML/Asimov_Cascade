# La Cascada de Asimov: Decaimiento Algorítmico de la Memoria Musical

## Abstract
Inspirado por la obra *Everywhere at the End of Time* de The Caretaker y el concepto de la "Cascada de Asimov" (el colapso de un modelo de IA que se entrena con sus propios datos sintéticos), este proyecto es un experimento de Recuperación de Información Musical (MIR). 
El sistema toma una composición original en formato MIDI y la somete a un bucle generativo de Cadenas de Markov. En cada iteración, el algoritmo intenta recomponer la obra basándose únicamente en el recuerdo de la iteración anterior, simulando un deterioro cognitivo progresivo (entropía algorítmica).

## Arquitectura del Sistema
1. **Parser Simbólico:** Uso de `music21` para extraer la secuencia cronológica de notas y acordes.
2. **Motor Generativo:** Modelo estocástico basado en Cadenas de Markov.
3. **Bucle de Decaimiento:** Automatización de N generaciones donde la salida G(n) se convierte en la entrada de G(n+1).

## Instalación
Para instalar las dependencias necesarias:
pip install -r requirements.txt