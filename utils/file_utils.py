import PyPDF2
from io import BytesIO
from pathlib import Path
import joblib
import streamlit as st

def extract_text_from_pdf(uploaded_file: BytesIO) -> str:
    """Extrai texto de arquivos PDF com tratamento de caracteres inválidos"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    # Remove caracteres problemáticos
                    page_text = page_text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                    text += page_text
            except Exception as e:
                st.warning(f"Erro ao extrair texto de uma página: {e}")
                continue
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
