import streamlit as st

def render_storytelling_page():
    """Renderiza a página de Storytelling do projeto."""
    st.title("📖 Storytelling do Projeto")
    st.markdown("""
    ### O Problema
    No dinâmico mercado de trabalho atual, recrutadores enfrentam um desafio monumental: analisar centenas, por vezes milhares, de currículos para cada vaga. Este processo manual não é apenas demorado e repetitivo, mas também está sujeito a vieses inconscientes que podem levar à exclusão de talentos promissores. Encontrar o candidato ideal numa pilha de documentos é como procurar uma agulha num palheiro.

    ### A Solução
    Esta ferramenta nasceu para revolucionar a triagem inicial de candidatos. Utilizando o poder do **Processamento de Linguagem Natural (NLP)** e de modelos de **Machine Learning**, a nossa aplicação transforma o processo de recrutamento. Ela lê e interpreta os currículos, compara-os de forma inteligente com os requisitos da vaga e gera um **score de compatibilidade** objetivo e baseado em dados.

    ### O Impacto
    O nosso objetivo é claro: **libertar o tempo e o talento dos recrutadores**. Ao automatizar a triagem, permitimos que os profissionais de RH se foquem no que realmente importa: as interações humanas, as entrevistas estratégicas e a construção de relações com os melhores talentos. A ferramenta não substitui o recrutador, mas sim potencia as suas capacidades, oferecendo uma análise objetiva que ajuda a construir equipas mais fortes e diversificadas.

    ### Jornada do Desenvolvimento
    1. **Análise do Problema**: Entendemos profundamente os desafios do recrutamento tradicional
    2. **Coleta de Dados**: Compilamos milhares de currículos e descrições de vagas reais
    3. **Modelagem**: Desenvolvemos e testamos diversos algoritmos de ML
    4. **Validação**: Parceria com RHs para testar e refinar o sistema
    5. **Implementação**: Criação desta plataforma intuitiva e poderosa

    ### Próximos Passos
    - Integração com APIs de plataformas de recrutamento
    - Análise de compatibilidade cultural
    - Sistema de recomendação de vagas para candidatos
    """)