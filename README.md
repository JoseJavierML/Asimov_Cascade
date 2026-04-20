# La Cascada de Asimov: Decaimiento Algorítmico de la Memoria Musical

## Abstract
Inspirado por la obra *Everywhere at the End of Time* de The Caretaker y el fenómeno técnico de la **"Cascada de Asimov"** (el colapso de un modelo de Inteligencia Artificial al retroalimentarse de sus propios datos sintéticos), este proyecto es un experimento de Recuperación de Información Musical (MIR) y Computación Estocástica.

El sistema toma una composición original (polifónica) y la somete a un bucle generativo de Cadenas de Markov. En cada iteración, el algoritmo intenta recomponer la obra basándose *únicamente* en el recuerdo de la iteración anterior, introduciendo una variable de entropía progresiva. El resultado es un simulador audible del deterioro cognitivo y el colapso del modelo.

## Arquitectura del Sistema

El motor consta de tres capas principales:

1. **Parser Simbólico Multipista (`music21`):** Diseccionamos archivos MIDI aislando instrumentos y leyendo su estructura armónica temporal.
2. **Motor Generativo Estocástico:** Uso de Cadenas de Markov de primer orden dinámicas. El sistema calcula las probabilidades de transición de estado (notas/acordes) basándose en la densidad de la memoria actual.
3. **Inyección de Entropía (El Factor de Olvido):** A medida que avanzan las generaciones, se inyecta una probabilidad $P(e)$ de "alucinación" (selección de un estado aleatorio fuera de la cadena lógica), simulando el ruido del colapso algorítmico.

## Las Fases del Colapso (The Caretaker Method)

Al ejecutar el motor, la obra atraviesa 15 fases de degradación:
* **Fases 1-5 (Lucidez Residual):** La música mantiene su coherencia armónica y rítmica. Pequeños errores de transposición o duración (duración dinámica proporcional).
* **Fases 6-10 (Confusión Post-Conciencia):** El motor multipista empieza a cruzar voces. Las progresiones lógicas se rompen. Las transiciones más probables sobreviven en bucles repetitivos.
* **Fases 11-15 (Entropía y Ruido Blanco):** La probabilidad de error supera a la memoria retenida. La obra se vuelve atonal, arrítmica y pierde su identidad original.

## Instalación y Uso

1. Clona este repositorio y crea un entorno virtual.
2. Instala las dependencias:
   pip install -r requirements.txt
3. Coloca tu archivo MIDI en la raíz con el nombre mi_obra.mid.
4. Ejecuta el motor:
    python motor_generativo.py