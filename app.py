import os
import tempfile
import streamlit as st
from memoria_base import extraer_adn_musical
from motor_generativo import (
    _escala_desde_secuencia,
    _temperatura_por_fase,
    construir_cadena_markov,
    generar_nueva_obra,
    sintetizar_audio_directo,
)

_TMPDIR = tempfile.mkdtemp(prefix="asimov_")

st.set_page_config(page_title="La Cascada de Asimov", page_icon="🎹", layout="wide")

st.title("🎹 La Cascada de Asimov")
st.markdown(
    "**Decaimiento Algorítmico de la Memoria Musical** — "
    "Sube tu composición y escucha cómo una IA la olvida progresivamente."
)
st.divider()

col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.subheader("📁 Obra original")
    uploaded_file = st.file_uploader(
        "Sube tu composición",
        type=["mid", "midi", "wav", "mp3"],
        help="MIDI recomendado. WAV/MP3 solo funcionan bien con melodías monofónicas (una sola voz).",
    )

with col_der:
    st.subheader("⚙️ Parámetros")
    num_fases    = st.slider("Fases de degradación", 3, 20, 10)
    temperatura_max = st.slider("Temperatura final", 0.50, 2.50, 1.80, 0.05, format="%.2f")
    orden_markov = st.select_slider("Orden Markov", [1, 2, 3, 4], value=2,
                                    help="Orden 1-2 para piezas cortas. 3-4 para piezas largas.")
    mostrar_onda = st.checkbox("Mostrar forma de onda", value=False)

st.divider()

if uploaded_file is None:
    st.info("⬆️ Sube un archivo para comenzar.")
    st.stop()

if not st.button("🚀 Iniciar Colapso del Modelo", use_container_width=True, type="primary"):
    st.stop()

ruta_temp = os.path.join(_TMPDIR, uploaded_file.name)
with open(ruta_temp, "wb") as fh:
    fh.write(uploaded_file.getbuffer())

es_audio = uploaded_file.name.lower().endswith((".wav", ".mp3"))
msg = "Analizando audio (puede tardar 30-60 s)…" if es_audio else "Extrayendo ADN del MIDI…"

with st.spinner(msg):
    try:
        memoria_actual = extraer_adn_musical(ruta_temp)
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
        st.stop()

for datos in memoria_actual.values():
    datos["escala_base"] = _escala_desde_secuencia(datos["secuencia"])

total_eventos = sum(len(v["secuencia"]) for v in memoria_actual.values())

with st.expander("🔍 Debug — primeros 10 eventos extraídos", expanded=False):
    for pista, datos in memoria_actual.items():
        st.write(f"**{pista}** — {len(datos['secuencia'])} eventos")
        st.write(datos["secuencia"][:10])

if total_eventos < 4:
    st.error(
        f"❌ Solo {total_eventos} evento(s) detectados. "
        "Usa MIDI, o un WAV/MP3 de un solo instrumento melódico (voz, flauta, violín)."
    )
    st.stop()

st.success(f"✅ {len(memoria_actual)} pista(s) · {total_eventos} eventos totales.")

if total_eventos < orden_markov * 20:
    st.warning(f"⚠️ Secuencia corta ({total_eventos} eventos) para orden {orden_markov}. Baja el orden a 1.")

st.markdown("### 🎼 Historial de Degradación")
progress_bar = st.progress(0)
status_text  = st.empty()

for f in range(1, num_fases + 1):
    temperatura = _temperatura_por_fase(f / num_fases, temperatura_max)
    status_text.text(f"Generando Fase {f}/{num_fases}  (Temperatura: {temperatura:.2f})…")

    nueva_memoria: dict = {}
    for pista, datos in memoria_actual.items():
        modelo    = construir_cadena_markov(datos["secuencia"], orden=orden_markov)
        modelo["escala_base"] = datos.get("escala_base") or _escala_desde_secuencia(datos["secuencia"])
        nueva_sec = generar_nueva_obra(modelo, len(datos["secuencia"]), temperatura, orden=orden_markov)
        nueva_memoria[pista] = {
            "instrumento": datos["instrumento"],
            "secuencia": nueva_sec,
            "escala_base": modelo["escala_base"],
        }

    nombre_salida = os.path.join(_TMPDIR, f"Fase_{f:02d}_Audio.wav")

    try:
        sintetizar_audio_directo(nueva_memoria, nombre_salida)
    except RuntimeError as e:
        st.error(f"❌ Error en síntesis de Fase {f}: {e}")
        st.stop()

    ratio    = f / num_fases
    etiqueta = "🟢 Lucidez Residual" if ratio <= 0.33 else ("🟡 Confusión" if ratio <= 0.66 else "🔴 Ruido Blanco")

    with st.expander(f"Fase {f:02d} — {temperatura:.2f}  {etiqueta}"):
        with open(nombre_salida, "rb") as wf:
            st.audio(wf.read(), format="audio/wav")

        if mostrar_onda:
            import soundfile as sf
            import numpy as np
            data, _ = sf.read(nombre_salida)
            paso = max(1, len(data) // 2000)
            st.line_chart({"onda": data[::paso]}, height=110)

        notas_unicas = len({ev[0] for pd in nueva_memoria.values() for ev in pd["secuencia"]})
        st.caption(f"Notas únicas: **{notas_unicas}** · Eventos: **{sum(len(v['secuencia']) for v in nueva_memoria.values())}**")

    memoria_actual = nueva_memoria
    progress_bar.progress(f / num_fases)

status_text.text("✅ Colapso completado.")
st.balloons()