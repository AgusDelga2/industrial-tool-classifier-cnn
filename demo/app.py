import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import threading
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Herramientas",
    page_icon="🛠️",
    layout="centered"
)

# ─────────────────────────────────────────────
# Estilos personalizados
# ─────────────────────────────────────────────
st.markdown("""
    <style>
        .stApp { background-color: #0f0f0f; color: #f0f0f0; }
        h1 { color: #f0f0f0; font-family: monospace; letter-spacing: 2px; }
        .resultado-box {
            padding: 16px 24px;
            border-radius: 10px;
            font-size: 1.4rem;
            font-weight: bold;
            text-align: center;
            margin-top: 12px;
            font-family: monospace;
        }
        .martillo        { background: #1a3d1a; border: 2px solid #00cc44; color: #00ff55; }
        .destornillador  { background: #1a1a3d; border: 2px solid #4488ff; color: #66aaff; }
        .desconocido     { background: #3d2a00; border: 2px solid #ff9900; color: #ffbb33; }
        .confianza-bar   { margin-top: 8px; font-size: 0.85rem; color: #aaaaaa; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Cargar modelo
# ─────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("modelo_herramientas.h5")

model = load_my_model()

# ─────────────────────────────────────────────
# Umbrales de confianza
# ─────────────────────────────────────────────
UMBRAL_ALTO = 0.60   # >= 0.60 → MARTILLO
UMBRAL_BAJO = 0.40   # <= 0.40 → DESTORNILLADOR
                     # entre ambos → DESCONOCIDO (baja confianza)

# ─────────────────────────────────────────────
# Clase del procesador de video (API actualizada)
# ─────────────────────────────────────────────
class VideoProcessor(VideoProcessorBase):
    """
    Procesa cada frame de la webcam:
    - Redimensiona a 224x224 y normaliza
    - Infiere con el modelo cargado
    - Dibuja el resultado sobre el frame (overlay)
    - Guarda el último resultado en atributos compartidos
      para que el hilo principal de Streamlit pueda leerlos
    """

    def __init__(self):
        self.label     = "Esperando..."
        self.confianza = 0.0
        self.lock      = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocesamiento en escala de grises (igual que en el entrenamiento)
        img_gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized    = cv2.resize(img_gray, (150, 150))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_reshaped   = np.expand_dims(img_normalized, axis=0)   # (1, 150, 150)
        img_reshaped   = np.expand_dims(img_reshaped,   axis=-1)  # (1, 150, 150, 1)

        # Inferencia
        raw       = model(img_reshaped, training=False)
        confianza = float(raw[0][0])

        # Clasificación con zona de desconfianza
        if confianza >= UMBRAL_ALTO:
            label = "MARTILLO"
            color = (0, 255, 0)        # Verde (BGR)
        elif confianza <= UMBRAL_BAJO:
            label = "DESTORNILLADOR"
            color = (255, 100, 0)      # Azul (BGR)
        else:
            label = "DESCONOCIDO"
            color = (0, 165, 255)      # Naranja (BGR)

        # Guardar resultado de forma thread-safe
        with self.lock:
            self.label     = label
            self.confianza = confianza

        # Overlay sobre el frame de video
        cv2.rectangle(img, (10, 10), (420, 70), (30, 30, 30), -1)
        cv2.putText(
            img,
            f"{label}  ({confianza:.2f})",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            2,
            cv2.LINE_AA
        )

        # IMPORTANTE: debe retornar av.VideoFrame, no un ndarray
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────
# UI principal
# ─────────────────────────────────────────────
st.title("🛠️  CLASIFICADOR DE HERRAMIENTAS")
st.caption("Mostrá un **martillo** o un **destornillador** a la cámara.")

col1, col2, col3 = st.columns(3)
col1.metric("Umbral Martillo",       f"≥ {UMBRAL_ALTO}")
col2.metric("Umbral Destornillador", f"≤ {UMBRAL_BAJO}")
col3.metric("Zona de desconfianza",  f"{UMBRAL_BAJO} – {UMBRAL_ALTO}")

st.divider()

# Refresca la UI cada 800ms para leer el último frame procesado
st_autorefresh(interval=800, limit=None, key="autorefresh")

# Placeholder para el resultado
resultado_placeholder = st.empty()

# Stream de webcam
ctx = webrtc_streamer(
    key="clasificador-herramientas",
    video_processor_factory=VideoProcessor,   # <- API actualizada
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Mostrar resultado del último frame procesado
if ctx.video_processor:
    with ctx.video_processor.lock:            # <- video_processor, no video_transformer
        label     = ctx.video_processor.label
        confianza = ctx.video_processor.confianza

    if label == "MARTILLO":
        resultado_placeholder.markdown(
            f'<div class="resultado-box martillo">🔨 {label}<br>'
            f'<span class="confianza-bar">Confianza: {confianza:.2f}</span></div>',
            unsafe_allow_html=True
        )
    elif label == "DESTORNILLADOR":
        resultado_placeholder.markdown(
            f'<div class="resultado-box destornillador">🪛 {label}<br>'
            f'<span class="confianza-bar">Confianza: {1 - confianza:.2f}</span></div>',
            unsafe_allow_html=True
        )
    else:
        resultado_placeholder.markdown(
            f'<div class="resultado-box desconocido">❓ {label}<br>'
            f'<span class="confianza-bar">Confianza insuficiente: {confianza:.2f} '
            f'(zona {UMBRAL_BAJO}–{UMBRAL_ALTO})</span></div>',
            unsafe_allow_html=True
        )

st.divider()
st.caption(
    "ℹ️ El resultado se muestra en el overlay del video en tiempo real. "
    "El panel de abajo refleja el último frame procesado."
)