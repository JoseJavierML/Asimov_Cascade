# La Cascada de Asimov: Decaimiento Algorítmico de la Memoria Musical

## Abstract
Inspirado por la obra *Everywhere at the End of Time* de The Caretaker y el fenómeno técnico de la **"Cascada de Asimov"** (el colapso de un modelo de Inteligencia Artificial al retroalimentarse de sus propios datos sintéticos), este proyecto es un experimento de Recuperación de Información Musical (MIR) y Computación Estocástica.

El sistema toma una composición original (polifónica, en formato MIDI o Audio Real) y la somete a un bucle generativo de Cadenas de Markov. En cada iteración, el algoritmo intenta recomponer la obra basándose *únicamente* en el recuerdo de la iteración anterior, introduciendo una variable de entropía progresiva. El resultado es un simulador audible del deterioro cognitivo y el colapso del modelo, accesible a través de una interfaz web interactiva.

## Arquitectura del Sistema

El motor consta de cuatro capas principales:

1. **Parser Universal (`music21` y `librosa`):** Procesamiento de datos simbólicos (MIDI multipista) y extracción de frecuencia fundamental (Pitch Tracking) para señales de audio digital (.wav, .mp3).
2. **Motor Generativo Estocástico:** Uso de Cadenas de Markov de primer orden dinámicas. El sistema calcula las probabilidades de transición de estado basándose en la densidad de la memoria actual.
3. **Inyección de Entropía (Factor de Olvido):** A medida que avanzan las generaciones, se inyecta una probabilidad de "alucinación" (selección de un estado fuera de la cadena lógica), simulando el ruido algorítmico.
4. **Síntesis y UI (`soundfile` y `Streamlit`):** Motor de síntesis aditiva en tiempo real que devuelve señales audibles (.wav) presentadas en un dashboard web interactivo.

## Las Fases del Colapso

Al ejecutar el motor, la obra atraviesa múltiples fases configurables por el usuario:
* **Lucidez Residual (Primer tercio):** La música mantiene su coherencia armónica y rítmica. Pequeños errores de transposición o duración.
* **Confusión Post-Conciencia (Segundo tercio):** El motor empieza a cruzar voces. Las progresiones lógicas se rompen en bucles repetitivos.
* **Ruido Blanco (Tercio final):** La entropía supera a la memoria retenida. La obra se vuelve atonal y pierde su identidad original.

---

## Despliegue

La forma recomendada y más estable de ejecutar este proyecto es mediante contenedores Docker, lo cual incluye todas las dependencias del sistema operativo (como `ffmpeg` para procesamiento de audio).

### Prerrequisitos
* Tener [Docker](https://www.docker.com/) instalado en tu máquina.

### Instalación y Ejecución
1. Clona este repositorio y navega a la carpeta del proyecto.
2. Construye la imagen de Docker:
   docker build -t asimov-cascade .
3. Levanta el contenedor exponiendo el puerto web:
   docker run -p 8501:8501 asimov-cascade
4. Abre tu navegador y entra en: http://localhost:8501

### Despliegue Normal
1. Crea un entorno virtual e instala las dependencias:
   pip install -r requirements.txt
2. Ejecuta la interfaz web:
   streamlit run app.py

## Licencia
Este proyecto se distribuye bajo la licencia GNU GPL v3. Queda permitido su uso, modificación y distribución para fines educativos, artísticos o comerciales, siempre que se reconozca la autoría original.