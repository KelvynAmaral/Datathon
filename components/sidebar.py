import streamlit as st

def render_sidebar():
    """Renderiza a barra lateral com navegação entre páginas"""
    with st.sidebar:
        st.title("Navegação")
        
        # Opções de navegação
        page_options = {
            "🔍 Análise": "Análise",
            "📈 Métricas": "Métricas",
            "📖 Storytelling": "Storytelling",
            "🛠️ Tecnologias": "Tecnologias"
        }
        
        # Cria os botões de navegação na sidebar
        selected_page = st.radio(
            "Selecione uma página:",
            options=list(page_options.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("**Sobre o Sistema**")
        st.markdown("""
        Esta aplicação utiliza Machine Learning 
        para otimizar a triagem de currículos, 
        comparando-os com os requisitos de uma vaga.
        """)
        
        st.markdown("---")
        st.markdown("Versão: 12.0 | 15/07/2025")
        
        return page_options[selected_page]