import streamlit as st
from datetime import datetime

def render_sidebar():
    """Renderiza a barra lateral da aplicação."""
    with st.sidebar:
        st.image("https://storage.googleapis.com/gemini-prod-us-west1-423907-8353/images/7a0402.png-ce9c0648-a0b8-45b7-8385-3dd46fef44e4", use_container_width=True)
        st.header("ℹ️ Sobre o Sistema")
        st.info("""
        Esta aplicação utiliza Machine Learning para otimizar a triagem de currículos,
        comparando-os com os requisitos de uma vaga.
        """)
        st.divider()
        st.markdown(f"**Versão:** 12.0 | {datetime.now().strftime('%d/%m/%Y')}")