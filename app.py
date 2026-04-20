import streamlit as st
import os
from memoria_base import extraer_adn_musical
from motor_generativo import construir_cadena_markov, generar_nueva_obra, sintetizar_audio_directo

st.set_page_config(page_title="Cascada de Asimov", page_icon="🎹", layout="centered")

st.title("La Cascada de Asimov")
st.markdown("""
**Decaimiento Algorítmico de la Memoria Musical** Sube tu composición original y escucha cómo una Inteligencia Artificial la olvida progresivamente a través de un bucle de entropía y Cadenas de Markov.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("1. Sube tu obra maestra", type=["mid", "midi", "wav", "mp3"])

with col2:
    num_fases = st.slider("2. Fases de degradación", min_value=3, max_value=20, value=10)
    p_error_max = st.slider("3. Nivel de Entropía final", min_value=0.1, max_value=0.8, value=0.35, help="35% significa que en la última fase habrá un 35% de notas aleatorias.")

if uploaded_file is not None:
    if st.button("Iniciar Colapso del Modelo", use_container_width=True):
        
        nombre_archivo_temp = uploaded_file.name
        with open(nombre_archivo_temp, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Archivo '{nombre_archivo_temp}' cargado. Extrayendo ADN musical...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            memoria_actual = extraer_adn_musical(nombre_archivo_temp)
            
            st.markdown("###Historial de Degradación")
            
            
            for f in range(1, num_fases + 1):
                p_error = (f / num_fases) * p_error_max
                status_text.text(f"Generando Fase {f}/{num_fases} (Entropía: {p_error:.1%})...")
                
                nueva_memoria = {}
                for pista, datos in memoria_actual.items():
                    modelo = construir_cadena_markov(datos['secuencia'])
                    nueva_sec = generar_nueva_obra(modelo, len(datos['secuencia']), p_error)
                    nueva_memoria[pista] = {
                        'instrumento': datos['instrumento'],
                        'secuencia': nueva_sec
                    }
                
                nombre_salida = f"Fase_{f:02d}_Audio.wav"
                sintetizar_audio_directo(nueva_memoria, nombre_salida)
            
                with st.expander(f"Fase {f} - Entropía: {p_error:.1%}"):
                    st.audio(nombre_salida, format='audio/wav')
                
                memoria_actual = nueva_memoria
                progress_bar.progress(f / num_fases)
                
            status_text.text("¡Colapso completado!")
            st.balloons() 
            
        except Exception as e:
            st.error(f"Ocurrió un error al procesar la música: {e}")