import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_resource
def setup_nltk():
    """Verifica e baixa os dados necessários do NLTK de forma segura."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    return set(stopwords.words('portuguese'))

def preprocessar_texto(texto: str, stopwords: set) -> str:
    """Limpa e pré-processa o texto para análise."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palavras = [palavra for palavra in texto.split() if palavra not in stopwords and len(palavra) >= 3]
    return ' '.join(palavras)

def extrair_competencias(texto_requisitos: str) -> set:
    """Extrai competências da caixa de texto, tratando termos com múltiplas palavras."""
    if not texto_requisitos:
        return set()
    return {comp.strip().lower() for comp in texto_requisitos.split(',') if comp.strip()}

def mapear_nivel(texto_cv: str, mapa: dict) -> int:
    """Função genérica para encontrar o maior nível de um mapa num texto."""
    if not texto_cv or pd.isna(texto_cv):
        return 0
    texto_lower = str(texto_cv).lower()
    niveis_encontrados = [valor for chave, valor in mapa.items() if chave in texto_lower]
    return max(niveis_encontrados) if niveis_encontrados else 0

def calcular_similaridade_texto(texto1: str, texto2: str, vectorizer: TfidfVectorizer) -> float:
    """Calcula a similaridade entre dois textos usando TF-IDF."""
    try:
        tfidf_matrix = vectorizer.transform([texto1, texto2])
        similaridade = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similaridade
    except Exception:
        return 0.0