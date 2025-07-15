import PyPDF2
from io import BytesIO
from pathlib import Path
import joblib
import streamlit as st

def extract_text_from_pdf(uploaded_file: BytesIO) -> str:
    """Extrai texto de arquivos PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except Exception as e:
        st.error(f"Erro ao ler PDF: {str(e)}")
        return ""

@st.cache_resource
def load_models():
    """Carrega os modelos ML salvos"""
    try:
        base_dir = Path(__file__).resolve().parent.parent
        model = joblib.load(base_dir / 'modelo_rf_final.pkl')
        scaler = joblib.load(base_dir / 'scaler_final.pkl')
        vectorizer = joblib.load(base_dir / 'tfidf_vectorizer.pkl')
        return model, scaler, vectorizer
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {str(e)}")
        st.stop()