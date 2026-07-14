# La Cascada de Asimov: Decaimiento Algorítmico de la Memoria Musical

## Abstract

Inspirado por la obra *Everywhere at the End of Time* de The Caretaker y el fenómeno técnico de la **"Cascada de Asimov"** (el colapso de un modelo de Inteligencia Artificial al retroalimentarse de sus propios datos sintéticos), este proyecto es un experimento de Recuperación de Información Musical (MIR) y Computación Estocástica.

El sistema toma una composición original (en formato MIDI o audio real) y la somete a un bucle generativo de Cadenas de Markov de orden N. En cada iteración, el algoritmo intenta recomponer la obra basándose *únicamente* en el recuerdo de la iteración anterior, introduciendo una variable de temperatura exponencial que simula el deterioro cognitivo. El resultado es un simulador audible del colapso del modelo, accesible a través de una interfaz web interactiva.

> **Nota sobre formatos de entrada:** Gracias al motor de separación armónica, el análisis de audio (WAV/MP3) es **polifónico**, siendo capaz de extraer e interpretar acordes de instrumentos como el piano directamente desde la onda de sonido.

---

## Arquitectura del Sistema

El motor consta de tres módulos principales:

### `memoria_base.py` — Parser Universal Polifónico
Extrae el ADN musical del archivo original como una secuencia de eventos.

- **Audio (`.wav`, `.mp3`):** Emplea separación de fuentes (HPSS) para aislar la armonía del ruido percusivo. Utiliza la Transformada de Q Constante (CQT) y detección de *onsets* para capturar hasta 4 notas simultáneas por evento, empaquetándolas en tuplas de acordes.
- **MIDI (`.mid`, `.midi`):** Usa `music21` para leer pistas preservando duraciones reales, acordes (tuplas) y silencios explícitos (`REST`).

### `motor_generativo.py` — Motor Generativo Estocástico
El núcleo cognitivo del sistema, reescrito para soportar armonía compleja. Compuesto por tres capas:

1. **Cadenas de Markov (Orden 1–4):** El modelo construye una tabla de transiciones donde los estados canónicos son acordes (tuplas). Con orden 2, el sistema recuerda los 2 acordes anteriores para decidir el siguiente, manteniendo la coherencia armónica inicial.
2. **Temperatura Exponencial (El Colapso):** Se abandona la probabilidad lineal por un sistema de *Temperatura* (Softmax) con sesgo tonal. En las primeras fases, la temperatura es casi nula (lucidez). A medida que avanzan las iteraciones, la temperatura sube con una curva exponencial hacia el límite de 3.0, forzando al modelo a ignorar la escala original y sumirse en el caos cromático.
3. **Síntesis FM (Piano Rhodes):** Las notas no son simples pitidos, sino que se renderizan utilizando un modelo de síntesis FM (frecuencia modulada) que simula el golpe físico, la resonancia y la calidez de un piano eléctrico clásico. Las voces de cada acorde se mezclan y normalizan dinámicamente para evitar distorsiones.

### `app.py` — Interfaz Web (Streamlit)
Dashboard interactivo con los siguientes controles:

| Parámetro | Rango | Descripción |
|---|---|---|
| Fases de degradación | 3–20 | Número de iteraciones del bucle generativo |
| Temperatura final | 0.5–3.0 | Límite máximo de caos en la iteración final |
| Orden de Markov | 1–4 | Contexto de memoria a corto plazo del modelo |
| Forma de onda | on/off | Visualización del espectro de audio por fase |

Cada fase genera un reproductor de audio independiente con estadísticas sobre el tamaño del vocabulario de acordes y la etiqueta de estado de colapso.

---

## Las Fases del Colapso

Al ejecutar el motor, la obra atraviesa tres fases sonoras identificadas visualmente:

- **🟢 Lucidez Residual:** La música mantiene la coherencia armónica y los acordes originales. Pequeños errores rítmicos o sutiles variaciones en la melodía, controlados por una temperatura mínima.
- **🟡 Confusión Post-Conciencia:** La curva exponencial empieza a notarse. El motor mezcla acordes lógicos en progresiones sin sentido. El ritmo se fractura y la obra se vuelve circular.
- **🔴 Colapso Estocástico (Ruido):** La temperatura alcanza su máximo. El sesgo tonal desaparece. El modelo genera estructuras disonantes, cromáticas y caóticas; la obra original es irreconocible.

---

## Despliegue

### Opción 1 — Docker (recomendado)

La forma más estable de ejecutar el proyecto, ya que incluye todas las dependencias del sistema operativo.

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

---

## Licencia

Este proyecto se distribuye bajo la licencia **GNU GPL v3**. Queda permitido su uso, modificación y distribución para fines educativos, artísticos o comerciales, siempre que se reconozca la autoría original.