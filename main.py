import sys
from pathlib import Path

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
        page_title="Sistema Inteligente de Triagem de Currículos",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# CSS para ocultar nomes das páginas padrão no menu lateral
    hide_page_names = """
        <style>
            [data-testid="stSidebarNav"] li {
                visibility: hidden;
                height: 0px;
                padding: 0px;
                margin: 0px;
            }

            /* Oculta imagem quebrada na sidebar */
            [data-testid="stSidebar"] img {
                display: none !important;
            }
        </style>
    """
    st.markdown(hide_page_names, unsafe_allow_html=True)
    
    # Renderiza a sidebar e obtém a página selecionada
    selected_page = render_sidebar()
    
    # Renderiza apenas o título principal
    st.title("📊 Sistema Inteligente de Triagem de Currículos")
    
    # Renderiza a página selecionada
    if selected_page == "Análise":
        render_main_page()
    elif selected_page == "Métricas":
        render_metrics_page()
    elif selected_page == "Storytelling":
        render_storytelling_page()
    elif selected_page == "Tecnologias":
        render_tech_page()

if __name__ == "__main__":
    main()