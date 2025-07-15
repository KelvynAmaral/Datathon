import streamlit as st

def render_metrics_page():
    """Renderiza a página com a explicação das métricas de avaliação."""
    st.title("📊 Métricas de Avaliação")
    st.markdown("""
    Esta seção detalha como cada candidato é avaliado, garantindo um processo transparente e baseado em dados.
    """)
    
    with st.expander("🔍 Como Funciona a Análise", expanded=True):
        st.markdown("""
        ### Processo de Avaliação em 4 Etapas:
        1. **Extração de Informações**: Leitura dos currículos e identificação de competências
        2. **Pré-processamento**: Limpeza e padronização do texto
        3. **Análise Comparativa**: Comparação com os requisitos da vaga
        4. **Classificação**: Cálculo do score final e status
        """)
    
    st.subheader("⚖️ Composição do Score Final")
    cols = st.columns(4)
    cols[0].metric("Match Técnico", "40%", "Peso no score")
    cols[1].metric("Modelo ML", "30%", "Peso no score")
    cols[2].metric("Similaridade", "20%", "Peso no score")
    cols[3].metric("Formação", "10%", "Peso no score")
    
    st.markdown("""
    ### Detalhamento das Métricas
    
    #### 1. Match Técnico (40%)
    - **O que mede:** Percentual de competências requeridas que foram encontradas no currículo
    - **Como calcular:** `(Nº competências encontradas) / (Total competências requeridas)`
    - **Exemplo:** Se a vaga pede 10 competências e o candidato tem 7 → 70%
    
    #### 2. Probabilidade do Modelo (30%)
    - **O que mede:** Confiança do modelo de Machine Learning na adequação do candidato
    - **Baseado em:** Random Forest treinado com dados históricos de contratações
    - **Features:** Match técnico, similaridade textual, aderência a requisitos
    
    #### 3. Similaridade Textual (20%)
    - **O que mede:** Similaridade semântica entre o currículo e a descrição da vaga
    - **Técnica:** TF-IDF + Cosine Similarity
    - **Considera:** Contexto, frequência de termos, relevância
    
    #### 4. Aderência Acadêmica (10%)
    - **O que mede:** Se a formação do candidato atende ao requisito mínimo
    - **Escala:** De 0% (não atende) a 100% (atende ou supera)
    """)
    
    st.subheader("📈 Interpretação dos Resultados")
    st.markdown("""
    | Status              | Score Range | Recomendação                          |
    |---------------------|-------------|---------------------------------------|
    | ✅ Recomendado      | 60-100%     | Forte aderência à vaga                |
    | 🟨 Potencial        | 40-59%      | Pode ser considerado com ressalvas    |
    | ❌ Baixa Aderência  | 0-39%       | Pouca compatibilidade com a vaga      |
    """)
    
    st.info("""
    💡 **Dica:** Utilize os filtros nas abas de resultados para explorar diferentes grupos de candidatos.
    Candidatos com status "Potencial" podem ter competências valiosas que justifiquem uma análise mais aprofundada.
    """)