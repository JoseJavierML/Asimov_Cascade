"""Capa visual de la interfaz de Streamlit.

Todo el adorno se inyecta como HTML/CSS dentro de la propia pagina, sin anadir
ninguna dependencia a requirements.txt. Asi el Dockerfile sigue siendo valido y
cualquiera que clone el proyecto obtiene el mismo aspecto sin configurar nada.

El fondo animado se hace con CSS puro y no con JavaScript a proposito: Streamlit
elimina los <script> aunque se inyecten con unsafe_allow_html=True, y meterlos
en un iframe (components.html) haria que el raton no llegase al canvas ni este
se dimensionase bien. Con capas de radial-gradient animadas el efecto se ve
igual, funciona en el documento principal y sobrevive a cada rerun.
"""
import streamlit as st

# Estrellas del campo: (x%, y%, radio px, color). Cada una es una capa de
# radial-gradient. Se generada una sola vez al importar el modulo.
def _campo_estelar(n_puntos: int = 42) -> str:
    import random

    generador = random.Random(1996)  # semilla fija: el fondo es siempre el mismo
    capas = []
    for i in range(n_puntos):
        x = generador.uniform(2, 98)
        y = generador.uniform(2, 98)
        r = generador.choice([1, 1, 1, 1.5, 1.5, 2])
        if i % 3 == 0:
            color = "rgba(168, 85, 247, 0.55)"    # violeta
        elif i % 3 == 1:
            color = "rgba(129, 140, 248, 0.5)"    # indigo claro
        else:
            color = "rgba(99, 102, 241, 0.45)"     # indigo
        capas.append(
            f"radial-gradient({r}px {r}px at {x:.1f}% {y:.1f}%, {color}, transparent)"
        )
    # La capa base: el resplandor superior de la web.
    capas.append("radial-gradient(circle at 50% 0%, #1a1a2e 0%, transparent 70%)")
    return ", ".join(capas)


def _tamanos(n_capas: int) -> str:
    return ", ".join(["1400px 1400px"] * (n_capas - 1) + ["100% 100%"])


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700&family=JetBrains+Mono:wght@100..800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', -apple-system, 'Segoe UI', sans-serif;
}

/* Fondo con profundidad: resplandor superior + campo de puntos a la deriva */
html { background-color: #030303; }
body {
    background-color: #030303;
    background-image: __CAMPO__;
    background-size: __TAMANOS__;
    background-attachment: fixed;
    animation: asimov-deriva 240s linear infinite;
}
@keyframes asimov-deriva {
    0%   { background-position: 0px 0px; }
    100% { background-position: 0px -1400px; }
}

.stApp { background: transparent; }
.block-container { padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1100px; }

/* Titulo */
.asimov-titulo {
    font-weight: 700;
    letter-spacing: -0.05em;
    line-height: 1.05;
    text-align: center;
    margin: 0.4rem 0 0.2rem 0;
}
.asimov-titulo .gradiente {
    background: linear-gradient(to right, #818cf8, #a855f7);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    padding-right: 0.05em;   /* compensa el tracking negativo: sin esto la
                                ultima letra se pinta de dos tonos */
}
.asimov-subtitulo {
    text-align: center;
    color: #6b7280;
    font-weight: 300;
    max-width: 40rem;
    margin: 0 auto 2.2rem auto;
    line-height: 1.7;
}
.asimov-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.35rem 1.1rem;
    border-radius: 9999px;
    background: rgba(99,102,241,0.10);
    border: 1px solid rgba(99,102,241,0.22);
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1.6rem;
}
.asimov-chip .asimov-sep { opacity: 0.5; }
.asimov-punto {
    width: 0.5rem; height: 0.5rem;
    border-radius: 50%;
    background: #6366f1;
    animation: asimov-latido 2s ease-in-out infinite;
}
@keyframes asimov-latido { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* Tarjetas */
.stButton > button {
    background: #4f46e5;
    border: 1px solid rgba(129,140,248,0.45);
    border-radius: 0.5rem;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.85rem 1rem;
    width: 100%;
    box-shadow: 0 0 30px rgba(99,102,241,0.28);
    transition: all 0.22s ease;
}
.stButton > button:hover {
    background: #6366f1;
    border-color: #a5b4fc;
    box-shadow: 0 0 44px rgba(99,102,241,0.45);
    transform: translateY(-1px);
}

/* Sliders */
[data-testid="stSlider"] [role="slider"] { background: #818cf8; }
[data-testid="stSlider"] [data-baseweb="track"] > div { background: #6366f1; }

/* Subtitulos de seccion */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #f3f4f6;
    font-weight: 500;
    letter-spacing: -0.02em;
}

/* Paneles desplegables */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 0.75rem;
    background: rgba(255,255,255,0.017);
    backdrop-filter: blur(10px);
}
[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
[data-testid="stExpander"]:hover { border-color: rgba(99,102,241,0.35); }

/* Avisos */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 0.75rem;
    backdrop-filter: blur(10px);
}

hr { border-color: rgba(255,255,255,0.07); }

/* Zona de arrastre de archivos */
[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(129,140,248,0.30);
    border-radius: 0.75rem;
    background: rgba(255,255,255,0.017);
}
[data-testid="stFileUploader"] section:hover { border-color: rgba(129,140,248,0.60); }

/* Respeta a quien pidio menos movimiento */
@media (prefers-reduced-motion: reduce) {
    .asimov-punto { animation: none; }
    body { animation: none; }
}
</style>
"""

def aplicar_tema(estado: str = "Estable", version: str = "v4.0") -> None:
    """Inyecta el tema y el fondo animado. Debe llamarse una vez, al inicio."""
    st.markdown(
        f"""
        <div style="text-align:center">
            <div class="asimov-chip">
                <span class="asimov-punto"></span>
                <span>Asimov Cascade</span>
                <span class="asimov-sep">·</span>
                <span>{version}</span>
                <span class="asimov-sep">·</span>
                <span>{estado}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    campo = _campo_estelar()
    css = _CSS.replace("__CAMPO__", campo).replace("__TAMANOS__", _tamanos(campo.count("radial-gradient")))
    st.markdown(css, unsafe_allow_html=True)


def cabecera(titulo: str, subtitulo: str) -> None:
    """Titulo principal con el degradado del proyecto."""
    st.markdown(
        f"""
        <h1 class="asimov-titulo">{titulo}<br><span style="color:#6b7280;font-weight:300">de </span><span class="gradiente">Asimov</span></h1>
        <p class="asimov-subtitulo">{subtitulo}</p>
        """,
        unsafe_allow_html=True,
    )