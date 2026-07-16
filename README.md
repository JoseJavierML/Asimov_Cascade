# La Cascada de Asimov: Decaimiento Algorítmico de la Memoria Musical

## Abstract

Inspirado por la obra *Everywhere at the End of Time* de The Caretaker y el fenómeno técnico de la **"Cascada de Asimov"** (el colapso de un modelo de Inteligencia Artificial al retroalimentarse de sus propios datos sintéticos), este proyecto es un experimento de Recuperación de Información Musical (MIR) y Computación Estocástica Concurrente.

El sistema toma una composición original (en formato MIDI o audio real polifónico), aísla sus instrumentos mediante redes neuronales y la somete a un bucle generativo de Cadenas de Markov multipista. En cada iteración, el algoritmo intenta recomponer la obra basándose *únicamente* en el recuerdo de la iteración anterior, introduciendo una variable de temperatura exponencial que simula el deterioro cognitivo. El resultado es un simulador audible del colapso del modelo, accesible a través de una interfaz web interactiva.

> **Nota sobre formatos de entrada:** El motor acústico procesa señales puras (WAV/MP3) utilizando un modelo avanzado de separación de fuentes de 6 pistas. Es capaz de diseccionar una mezcla completa en capas individuales (piano, bajo, batería, guitarra, etc.), extrayendo la armonía, el ritmo y la dinámica humana de cada una por separado.

---

## Arquitectura del Sistema

El motor consta de tres módulos principales:

### `memoria_base.py` — Triaje Acústico y Extracción Multidimensional
Procesa el audio de entrada y construye la memoria base estandarizada.

- **Separación de Fuentes (Stem Separation):** Utiliza el modelo `htdemucs_6s` de Meta para aislar quirúrgicamente 6 pistas (Voces, Batería, Bajo, Piano, Guitarra y Otros). Las pistas retenidas se procesan de forma independiente.
- **Oído Biónico y DSP:** Aplica recorte automático de silencios y ajustes de alta sensibilidad (umbral -55dB, delta 0.03) para no perder dinámicas tenues (como cajas de música o subgraves). Combina Separación Armónica/Percusiva (HPSS) y la Transformada de Q Constante (CQT).
- **Super-Estados de Memoria:** El sistema ya no extrae notas aisladas, sino eventos tridimensionales exactos. Cada estado codificado contiene: `(Tupla_de_Acorde, Duración_Cuantizada, Velocidad_RMS)`. El reloj metronómico soporta tresillos y puntillos, y las dinámicas se agrupan en 5 niveles para preservar el "groove" y la intención física.

### `motor_generativo.py` — Clúster Estocástico y Síntesis Multitímbrica
El núcleo cognitivo del sistema, rediseñado para soportar caos concurrente.

1. **Clúster de Markov en Paralelo:** Instancia un cerebro de Markov independiente para cada instrumento (stem). Cada cadena tiene su propio proceso de olvido y degradación basado en los Super-Estados.
2. **Temperatura Exponencial (El Colapso):** Aplica un sistema de Temperatura (Softmax) con sesgo tonal. De la lucidez absoluta en las primeras fases (donde el tiempo base se limita estrictamente a la duración original), la curva escala hacia un límite caótico, forzando disonancias y la fractura de la percepción temporal.
3. **Motor de Síntesis Multitímbrica (Stem-Aware):** Las secuencias generadas se renderizan usando perfiles sonoros específicos según el instrumento:
   - `piano`: Síntesis FM (Piano Rhodes eléctrico).
   - `bass`: Sintetizador de ondas cuadradas / subgraves.
   - `drums`: Generador de transitorios y ráfagas de ruido blanco filtrado.
   - `guitar`: Sintetizador de ondas diente de sierra (Pluck/Sawtooth).
   - `other`: Sintetizador de Pad / Campana de cristal (Ondas senoidales puras).
4. **Bus Maestro y Limitador:** Las ondas de todos los instrumentos generados se suman en un único canal. Se aplica normalización y limitación estricta (*soft-clipping*) para evitar distorsiones digitales cuando las frecuencias chocan masivamente en las fases finales del colapso.

### `app.py` — Interfaz Web (Streamlit)
Dashboard interactivo con los siguientes controles:

| Parámetro | Rango | Descripción |
|---|---|---|
| Fases de degradación | 3–20 | Número de iteraciones del bucle generativo multipista |
| Temperatura final | 0.5–3.0 | Límite máximo de caos en la iteración final |
| Orden de Markov | 1–4 | Contexto de memoria a corto plazo del modelo |
| Forma de onda | on/off | Visualización del espectro de audio masterizado por fase |

---

## Las Fases del Colapso

Al ejecutar el motor, la obra atraviesa tres fases sonoras identificadas visualmente:

- **🟢 Lucidez Residual:** La banda mantiene la cohesión. Se respetan los acordes, los tiempos y las dinámicas originales. La separación de instrumentos es clara y la temperatura es mínima.
- **🟡 Confusión Post-Conciencia:** Los cerebros individuales empiezan a desincronizarse. El bajo pierde su relación con el piano, el ritmo de la batería se vuelve aleatorio. Mezcla de progresiones lógicas en contextos absurdos.
- **🔴 Colapso Estocástico (Ruido):** La temperatura aplasta la matriz de transición. Los sintetizadores disparan super-estados al azar chocando contra el limitador del Bus Maestro. Surge un muro de sonido disonante donde la obra original desaparece en el vacío.

---

## Despliegue

### Requisito Crítico del Sistema (FFMPEG)
Para que el modelo de separación neuronal (Demucs) y las librerías de audio (Pydub/Librosa) funcionen, es **obligatorio** que el sistema operativo tenga `ffmpeg` instalado en sus variables de entorno (`PATH`).

- **Windows:** `winget install ffmpeg` (Requiere reiniciar la terminal/IDE tras la instalación).
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

### Opción 1 — Docker (recomendado)

La forma más estable de ejecutar el proyecto, ya que incluye FFMPEG y todas las dependencias aisladas.

**Prerrequisitos:** tener [Docker](https://www.docker.com/) instalado.

```bash
# Construir la imagen
docker build -t asimov-cascade .

# Levantar el contenedor
docker run -p 8501:8501 asimov-cascade
```

Abre tu navegador en: **http://localhost:8501**

### Opción 2 — Entorno local

*Nota: La primera vez que se ejecute el procesamiento de un archivo de audio, el sistema descargará los pesos del modelo `htdemucs_6s` desde los servidores de Meta. Esto puede tardar un par de minutos dependiendo de la conexión.*

```bash
# Crear entorno virtual (recomendado)
python -m venv env
source env/bin/activate        # Linux/macOS
env\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

---

## Licencia

Este proyecto se distribuye bajo la licencia **GNU GPL v3**. Queda permitido su uso, modificación y distribución para fines educativos, artísticos o comerciales, siempre que se reconozca la autoría original.