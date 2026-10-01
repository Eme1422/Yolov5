from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

# Configuración inicial de la interfaz
st.set_page_config(
    page_title="VisionAI - Visión Artificial",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados con temática Amarillo Cálido / Ámbar
st.markdown("""
    <style>
    /* Fondo principal en tono crema/amarillo tenue */
    .stApp {
        background-color: #FFFDF0;
    }
    
    /* Barra lateral cálida */
    [data-testid="stSidebar"] {
        background-color: #FEF3C7;
        border-right: 1px solid #FDE68A;
    }
    
    /* Encabezados y títulos */
    h1, h2, h3 {
        color: #78350F !important;
        font-weight: 700 !important;
    }

    /* Estilo de sliders y controles */
    .stSlider > div > div > div > div {
        background-color: #D97706 !important;
    }

    /* Botones primarios */
    .stButton>button {
        background-color: #F59E0B !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(217, 119, 6, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #D97706 !important;
        box-shadow: 0 4px 8px rgba(217, 119, 6, 0.35) !important;
    }

    /* Cajas con borde para estructurar componentes */
    [data-testid="stVerticalBlock"] > div > [data-testid="stBlock"] {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Carga en caché de la red neuronal
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Fallo crítico al inicializar la arquitectura del modelo: {str(e)}")
        return None

# Encabezado principal
st.title("⚡ Visión Inteligente en Tiempo Real")
st.caption("Reconocimiento automatizado de elementos visuales impulsado por redes neuronales YOLOv5.")
st.divider()

with st.spinner("Inicializando motor de visión artificial..."):
    model = load_model()

if model:
    # Ajustes en la barra lateral
    with st.sidebar:
        st.image("https://img.icons8.com/isometric/512/compact-camera.png", width=70)
        st.title("Panel de Control")
        st.subheader("Sensibilidad y Calibración")
        
        conf_threshold = st.slider("Certeza Mínima (Confidence)", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Solapamiento (IoU Threshold)", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Límite de Objetos por Toma", 10, 2000, 1000, 10)
        
        st.divider()
        st.info("💡 **Recomendación:** Ajusta la 'Certeza Mínima' si el modelo pasa por alto elementos o genera falsos positivos.")

    # Captura de imagen
    st.subheader("📸 Captura de Imagen")
    picture = st.camera_input("Toma una captura para analizar la escena", key="camera")

    if picture:
        bytes_data = picture.getvalue()

        # Decodificación y conversión BGR para el pipeline de detección
        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]

        with st.spinner("Escaneando patrones y procesando detección..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det)
                )
            except Exception as e:
                st.error(f"Error durante el escaneo visual: {str(e)}")
                st.stop()

        result        = results[0]
        boxes         = result.boxes
        annotated     = result.plot()
        annotated_rgb = annotated[:, :, ::-1]  # BGR a RGB

        st.divider()

        # Mostrar métricas generales rápidas
        if boxes is not None and len(boxes) > 0:
            total_detectado = len(boxes)
            label_names = model.names
            categorias = [label_names[int(b.cls.item())] for b in boxes]
            cat_mas_frecuente = max(set(categorias), key=categorias.count)

            m1, m2, m3 = st.columns(3)
            m1.metric("Objetos Identificados", total_detectado)
            m2.metric("Elemento Predominante", cat_mas_frecuente.capitalize())
            m3.metric("Confianza Máxima", f"{float(boxes.conf.max().item()):.2%}")

            st.divider()

        # Columnas de resultados visuales y tabulares
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader("🎯 Resultado Visual")
                st.image(annotated_rgb, use_container_width=True, caption="Objetos demarcados mediante bounding boxes.")

        with col2:
            with st.container(border=True):
                st.subheader("📊 Métricas de Clasificación")
                if boxes is not None and len(boxes) > 0:
                    label_names    = model.names
                    category_count = {}
                    category_conf  = {}

                    for box in boxes:
                        cat  = int(box.cls.item())
                        conf = float(box.conf.item())
                        category_count[cat] = category_count.get(cat, 0) + 1
                        category_conf.setdefault(cat, []).append(conf)

                    data = [
                        {
                            "Categoría": label_names[cat].capitalize(),
                            "Unidades": count,
                            "Precisión Promedio": f"{np.mean(category_conf[cat]):.1%}"
                        }
                        for cat, count in category_count.items()
                    ]

                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.bar_chart(df.set_index("Categoría")["Unidades"])
                else:
                    st.warning("⚠️ No se identificaron objetos que cumplan con los parámetros actuales.")
                    st.caption("Intenta reducir el nivel de 'Certeza Mínima' en la barra lateral para detectar más elementos.")
else:
    st.error("No fue posible activar la red neuronal. Confirma que el paquete `ultralytics` y sus dependencias estén instalados correctamente.")
    st.stop()

# Pie de página
st.divider()
st.caption("⚡ **Visión Inteligente** — Sistema de detección analítico impulsado por YOLOv5, PyTorch y Streamlit.")
