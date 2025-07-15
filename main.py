import sys
from pathlib import Path

# Adiciona o diretório raiz ao path do Python
sys.path.append(str(Path(__file__).parent))

import streamlit as st
from components.sidebar import render_sidebar
from pages.analysis_page import render_main_page
from pages.metrics_page import render_metrics_page
from pages.storytelling import render_storytelling_page
from pages.tech_page import render_tech_page

def main():
    """Função principal que configura e executa a aplicação"""
    st.set_page_config(
        page_title="Sistema de Triagem de CVs",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    render_sidebar()
    st.title("📊 Sistema Inteligente de Triagem de Currículos")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Análise", "📈 Métricas", "📖 Storytelling", "🛠️ Tecnologias"
    ])
    
    with tab1:
        render_main_page()
    with tab2:
        render_metrics_page()
    with tab3:
        render_storytelling_page()
    with tab4:
        render_tech_page()

if __name__ == "__main__":
    main()