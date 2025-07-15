import streamlit as st

def render_tech_page():
    """Renderiza a página com as tecnologias utilizadas no projeto."""
    st.title("🛠️ Tecnologias Utilizadas")
    st.markdown("Esta aplicação foi construída com um conjunto de tecnologias modernas e robustas do ecossistema Python, focadas em ciência de dados e desenvolvimento web.")
    
    st.subheader("💻 Interface e Visualização")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        #### Streamlit
        - Framework principal para UI web
        - Permite criar apps interativas rapidamente
        - Ideal para protótipos e produção
        """)
    with cols[1]:
        st.markdown("""
        #### Plotly
        - Visualizações interativas
        - Gráficos profissionais
        - Integração perfeita com Streamlit
        """)
    with cols[2]:
        st.markdown("""
        #### Pandas
        - Manipulação de dados tabulares
        - Análise eficiente
        - Integração com visualizações
        """)

    st.subheader("🧠 Machine Learning e NLP")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        #### Scikit-learn
        - Modelos de ML prontos
        - Pré-processamento de dados
        - Métricas de avaliação
        """)
    with cols[1]:
        st.markdown("""
        #### NLTK
        - Processamento de linguagem natural
        - Tokenização
        - Remoção de stopwords
        """)
    with cols[2]:
        st.markdown("""
        #### Joblib
        - Serialização de modelos
        - Armazenamento eficiente
        - Carregamento rápido
        """)

    st.subheader("📁 Processamento de Arquivos")
    st.markdown("""
    #### PyPDF2
    - Extração de texto de PDFs
    - Manipulação de documentos
    - Leitura em memória
    """)

    st.divider()
    st.subheader("🔧 Arquitetura do Sistema")
    st.image("https://storage.googleapis.com/gemini-prod-us-west1-423907-8353/images/7a0402.png-ce9c0648-a0b8-45b7-8385-3dd46fef44e4", 
             caption="Diagrama de Arquitetura Simplificado", width=600)