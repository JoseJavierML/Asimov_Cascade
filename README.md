# La Cascada de Asimov: Decaimiento Algorítmico de la Memoria Musical

## Abstract

Inspirado por la obra *Everywhere at the End of Time* de The Caretaker y el fenómeno técnico de la **"Cascada de Asimov"** (el colapso de un modelo de Inteligencia Artificial al retroalimentarse de sus propios datos sintéticos), este proyecto es un experimento de Recuperación de Información Musical (MIR) y Computación Estocástica.

El sistema toma una composición original (en formato MIDI o audio real) y la somete a un bucle generativo de Cadenas de Markov de orden N. En cada iteración, el algoritmo intenta recomponer la obra basándose *únicamente* en el recuerdo de la iteración anterior, introduciendo una variable de entropía progresiva que afecta tanto a las notas como a las duraciones. El resultado es un simulador audible del deterioro cognitivo y el colapso del modelo, accesible a través de una interfaz web interactiva.

> **Nota sobre formatos de entrada:** el análisis de audio WAV/MP3 funciona bien con melodías monofónicas (una sola voz: flauta, violín, voz solista). Para composiciones polifónicas con varios instrumentos, se recomienda usar MIDI.

---

## Arquitectura del Sistema

El motor consta de tres módulos principales:

### `memoria_base.py` — Parser Universal
Extrae el ADN musical del archivo original como una secuencia de eventos `(nota, duración)`.

- **MIDI** (`.mid`, `.midi`): usa `music21` para leer pistas multipista preservando duraciones reales, acordes y silencios (`REST`).
- **Audio** (`.wav`, `.mp3`): usa `librosa` con el algoritmo **pYIN** para pitch tracking frame a frame. Los frames consecutivos en la misma nota se agrupan mediante run-length encoding para obtener la duración real de cada nota. Se filtran eventos de duración inferior a 80 ms como ruido.

### `motor_generativo.py` — Motor Generativo Estocástico
Núcleo del sistema. Compuesto por tres capas:

1. **Cadenas de Markov de orden N configurable** (1–4): el modelo construye una tabla de transiciones trabajando únicamente con strings de notas. Las duraciones se gestionan en un diccionario separado para evitar colisiones de tipos en las claves. Con orden 2, el sistema recuerda los 2 estados anteriores para decidir el siguiente, produciendo coherencia melódica durante más tiempo antes del colapso.

2. **Inyección de entropía dual (Factor de Olvido):** en cada iteración se inyecta una probabilidad de "alucinación" que crece linealmente con las fases. Afecta a las notas (selección aleatoria fuera de la cadena) y, con un factor 0.6×, a las duraciones (preservando el ritmo durante más tiempo que la melodía, en analogía con el deterioro cognitivo real).

3. **Síntesis aditiva con armónicos:** cada nota se sintetiza sumando 8 parciales con amplitudes según el perfil del instrumento (piano, cuerdas, órgano). Incluye envoltura ADSR por nota. Se mezclan todas las pistas en un único WAV normalizado sin clipping.

### `app.py` — Interfaz Web (Streamlit)
Dashboard interactivo con los siguientes controles:

| Parámetro | Rango | Descripción |
|---|---|---|
| Fases de degradación | 3–20 | Número de iteraciones del bucle generativo |
| Entropía final | 5–80 % | Porcentaje de alucinaciones en la última fase |
| Orden de Markov | 1–4 | Contexto de memoria del modelo |
| Forma de onda | on/off | Visualización del audio por fase |

Cada fase genera un reproductor de audio independiente con estadísticas (notas únicas, eventos totales) y una etiqueta de estado del colapso.

---

## Las Fases del Colapso

Al ejecutar el motor, la obra atraviesa tres fases identificadas visualmente en la interfaz:

- **🟢 Lucidez Residual (primer tercio):** la música mantiene coherencia armónica y rítmica. Pequeños errores de transposición o duración apenas perceptibles.
- **🟡 Confusión Post-Conciencia (segundo tercio):** el motor empieza a cruzar voces. Las progresiones lógicas se rompen en bucles repetitivos. El ritmo empieza a disociarse de la melodía.
- **🔴 Ruido Blanco (tercio final):** la entropía supera la memoria retenida. La obra se vuelve atonal y pierde su identidad original.

---

## Despliegue

### Opción 1 — Docker (recomendado)

La forma más estable de ejecutar el proyecto, ya que incluye todas las dependencias del sistema operativo (como `ffmpeg`).

**Prerrequisitos:** tener [Docker](https://www.docker.com/) instalado.

```bash
# Construir la imagen
docker build -t asimov-cascade .

# Levantar el contenedor
docker run -p 8501:8501 asimov-cascade
```

Abre tu navegador en: **http://localhost:8501**

---

### Opción 2 — Entorno local

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

**Dependencias principales:**

| Librería | Uso |
|---|---|
| `streamlit` | Interfaz web |
| `music21` | Parsing MIDI multipista |
| `librosa` | Pitch tracking y análisis de audio |
| `soundfile` | Escritura de archivos WAV |
| `numpy` | Síntesis y operaciones numéricas |

---

## Licencia

Este proyecto se distribuye bajo la licencia **GNU GPL v3**. Queda permitido su uso, modificación y distribución para fines educativos, artísticos o comerciales, siempre que se reconozca la autoría original.