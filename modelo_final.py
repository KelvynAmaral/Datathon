import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.preprocessing import MinMaxScaler
import PyPDF2
from io import BytesIO
from datetime import datetime
from pathlib import Path

# --- CONFIGURAÇÃO INICIAL E CONSTANTES ---

MAPA_NIVEL_PROFISSIONAL = {
    'aprendiz': 1, 'trainee': 2, 'auxiliar': 3, 'assistente': 4,
    'técnico de nível médio': 5, 'júnior': 5.5, 'analista': 6,
    'pleno': 7, 'supervisor': 7, 'líder': 7.5, 'sênior': 8,
    'especialista': 9, 'coordenador': 9, 'gerente': 10
}

# --- FUNÇÕES DE SETUP E CARREGAMENTO ---

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

STOPWORDS_PT = setup_nltk()

@st.cache_resource
def load_models():
    """Carrega os modelos de forma robusta, evitando erros de caminho."""
    try:
        base_dir = Path(__file__).resolve().parent
        model = joblib.load(base_dir / 'modelo_rf_final.pkl')
        scaler = joblib.load(base_dir / 'scaler_final.pkl')
        vectorizer = joblib.load(base_dir / 'tfidf_vectorizer.pkl')
        return model, scaler, vectorizer
    except FileNotFoundError as e:
        st.error(f"Erro de Ficheiro Não Encontrado: '{e.filename}'.")
        st.warning("Certifique-se de que os ficheiros de modelo (.pkl) estão na mesma pasta que a aplicação.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar os modelos: {str(e)}")
        st.stop()

# --- FUNÇÕES DE PROCESSAMENTO DE DADOS ---

def extract_text_from_pdf(uploaded_file: BytesIO) -> str:
    """Extrai texto de um ficheiro PDF em memória."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except Exception as e:
        st.error(f"Erro ao ler o ficheiro PDF: {str(e)}")
        return ""

def preprocessar_texto(texto: str) -> str:
    """Limpa e pré-processa o texto para análise."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palavras = [palavra for palavra in texto.split() if palavra not in STOPWORDS_PT and len(palavra) >= 3]
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

def calcular_status(score: float) -> tuple[str, str]:
    """Calcula o status e a cor correspondente com base no score."""
    if score >= 0.6:
        return "✅ Apto (Alto)", "green"
    elif score >= 0.4:
        return "🟨 Em análise (Médio)", "orange"
    else:
        return "❌ Não apto (Baixo)", "red"

# --- FUNÇÕES DE RENDERIZAÇÃO DE PÁGINAS E COMPONENTES ---

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
        st.markdown(f"**Versão:** 10.0 | {datetime.now().strftime('%d/%m/%Y')}")

def render_results(df_resultados, detalhes_candidatos, job_title):
    """Renderiza os resultados da análise em múltiplas abas."""
    st.success(f"✅ Análise concluída para {len(df_resultados)} candidatos para a vaga de **{job_title}**!")
    
    tab_dashboard, tab_ranking, tab_individual, tab_export = st.tabs(["🏆 Dashboard", "📊 Ranking Geral", "👤 Análise Individual", "📤 Exportar"])
    
    with tab_dashboard:
        st.header("Dashboard da Análise")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Candidatos Analisados", len(df_resultados))
            melhor_score = df_resultados['Score Combinado'].max()
            st.metric("Melhor Score Encontrado", f"{melhor_score:.1%}")
        with col2:
            status_counts = df_resultados['Status'].value_counts()
            st.dataframe(status_counts, use_container_width=True)
        st.subheader("Distribuição de Status")
        st.bar_chart(status_counts)

    with tab_ranking:
        st.header("Visão Geral dos Candidatos")
        st.dataframe(
            df_resultados[["ID", "Nome", "Score Combinado", "Status", "Probabilidade", "Match"]],
            column_config={
                "Score Combinado": st.column_config.ProgressColumn("Score", format="%.1f%%", min_value=0, max_value=1),
                "Probabilidade": st.column_config.ProgressColumn("Prob.", format="%.1f%%", min_value=0, max_value=1),
                "Match": st.column_config.ProgressColumn("Match", format="%.1f%%", min_value=0, max_value=1),
            },
            hide_index=True,
            use_container_width=True
        )

    with tab_individual:
        st.header("Análise Detalhada por Candidato")
        for detalhe in detalhes_candidatos:
            with st.container(border=True):
                score_combinado = df_resultados.loc[df_resultados['ID'] == detalhe['ID'], 'Score Combinado'].iloc[0]
                st.subheader(f"� {detalhe['Nome']} (Score: {score_combinado:.1%})")
                
                st.markdown("**Métricas Principais:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Probabilidade do Modelo", f"{detalhe['Probabilidade']:.1%}")
                with col2:
                    st.metric("Match de Competências", f"{detalhe['Match']:.1%}")
                with col3:
                    st.metric("Score Combinado", f"{score_combinado:.1%}")

                st.markdown("**Aderência aos Requisitos:**")
                col_ader1, col_ader2, col_ader3 = st.columns(3)
                with col_ader1:
                    st.metric("Nível Académico", f"{detalhe['Aderência Acadêmica']*100:.0f}%", help="100% = Atende ou supera o requisito.")
                with col_ader2:
                    st.metric("Nível de Inglês", f"{detalhe['Aderência Inglês']*100:.0f}%", help="100% = Atende ou supera o requisito.")
                with col_ader3:
                    st.metric("Nível de Espanhol", f"{detalhe['Aderência Espanhol']*100:.0f}%", help="100% = Atende ou supera o requisito.")

                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.markdown("**Competências encontradas:**")
                    st.success(detalhe["TermosEncontrados"] or "Nenhuma competência encontrada.", icon="✅")
                with col_comp2:
                    st.markdown("**Principais competências em falta:**")
                    st.warning(detalhe["TermosFaltantes"] or "Nenhuma competência em falta.", icon="⚠️")
                
                st.markdown("**Trecho processado do currículo:**")
                st.text_area("Texto Analisado", value=detalhe["TextoProcessado"], height=150, disabled=True, key=f"texto_{detalhe['ID']}")
    
    with tab_export:
        st.header("Download dos Resultados da Análise")
        st.markdown("Faça o download dos dados da análise nos formatos CSV.")
        
        with st.container(border=True):
            df_detalhes_export = pd.DataFrame(detalhes_candidatos)
            df_export_completo = pd.merge(df_resultados, df_detalhes_export, on="ID")
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_resumo = df_resultados.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
                st.download_button("📊 Exportar Resumo CSV", csv_resumo, f"resumo_candidatos_{job_title.replace(' ', '_')}.csv", "text/csv", use_container_width=True)
            with col_exp2:
                csv_detalhes = df_export_completo.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
                st.download_button("📝 Exportar Detalhes CSV", csv_detalhes, f"detalhes_candidatos_{job_title.replace(' ', '_')}.csv", "text/csv", use_container_width=True)

def render_main_page():
    """Renderiza a página principal da ferramenta de análise."""
    st.header("🎯 Ferramenta de Análise")
    with st.container(border=True):
        with st.form("vaga_form"):
            st.subheader("📝 Informações da Vaga")
            job_title = st.text_input("Título da Vaga*", placeholder="Ex: Engenheiro de Dados Sênior")
            job_requirements = st.text_area("Competências Técnicas Requeridas*", placeholder="Ex: Python, SQL, Power BI, Machine Learning...", height=150, help="Liste as habilidades-chave (incluindo as com múltiplas palavras) separadas por vírgula.")
            col1, col2 = st.columns(2)
            with col1:
                job_academic_level = st.selectbox("Nível Académico Requerido*", ["Ensino Médio", "Ensino Técnico", "Ensino Superior", "Pós-Graduação", "Mestrado", "Doutorado"])
                job_english = st.selectbox("Inglês Exigido*", ["Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"])
            with col2:
                job_professional_level = st.selectbox("Nível Profissional*", list(MAPA_NIVEL_PROFISSIONAL.keys()))
                job_spanish = st.selectbox("Espanhol Exigido", ["Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"])
            st.subheader("👤 Currículos para Análise")
            uploaded_files = st.file_uploader("Selecione os currículos em PDF*", type=["pdf"], accept_multiple_files=True)
            submitted = st.form_submit_button("🚀 Analisar Candidatos", use_container_width=True)

    if "resultados_df" in st.session_state and not submitted:
        render_results(st.session_state.resultados_df, st.session_state.detalhes_candidatos, st.session_state.job_title)

    if submitted:
        if not all([job_title, job_requirements, uploaded_files]):
            st.error("🚨 Por favor, preencha todos os campos obrigatórios (*) e envie pelo menos um currículo.")
            st.stop()
        
        model, scaler, tfidf_vectorizer = load_models()
        resultados, detalhes_candidatos = [], []
        termos_vaga = extrair_competencias(job_requirements)
        req_preprocessados_tfidf = preprocessar_texto(" ".join(termos_vaga))
        mapa_academico = {"ensino médio": 1, "ensino técnico": 1.5, "ensino superior": 2, "pós-graduação": 3, "mestrado": 4, "doutorado": 5}
        mapa_idioma = {"nenhum": 0, "básico": 1, "intermediário": 2, "avançado": 3, "fluente": 4}
        nivel_academico_vaga = mapear_nivel(job_academic_level, mapa_academico)
        nivel_ingles_vaga = mapear_nivel(job_english, mapa_idioma)
        nivel_espanhol_vaga = mapear_nivel(job_spanish, mapa_idioma)
        nivel_profissional_vaga = MAPA_NIVEL_PROFISSIONAL.get(job_professional_level, 0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            with st.spinner(f"Processando {uploaded_file.name}..."):
                cv_text_raw = extract_text_from_pdf(BytesIO(uploaded_file.getvalue()))
                if not cv_text_raw:
                    st.warning(f"Não foi possível extrair texto de {uploaded_file.name}.")
                    continue
                cv_preprocessado_completo = preprocessar_texto(cv_text_raw)
                termos_encontrados = {term for term in termos_vaga if term in cv_text_raw.lower()}
                match_percent = len(termos_encontrados) / len(termos_vaga) if termos_vaga else 0
                try:
                    tfidf_matrix = tfidf_vectorizer.transform([req_preprocessados_tfidf, cv_preprocessado_completo])
                    similaridade = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                except Exception:
                    similaridade = 0.0
                nivel_academico_candidato = mapear_nivel(cv_text_raw, mapa_academico)
                aderencia_academica = 1.0 if nivel_academico_candidato >= nivel_academico_vaga else 0.0
                nivel_ingles_candidato = mapear_nivel(cv_text_raw, mapa_idioma)
                aderencia_ingles = 1.0 if nivel_ingles_candidato >= nivel_ingles_vaga else 0.0
                nivel_espanhol_candidato = mapear_nivel(cv_text_raw, mapa_idioma)
                aderencia_espanhol = 1.0 if nivel_espanhol_candidato >= nivel_espanhol_vaga else 0.0
                
                features = np.array([[
                    match_percent, similaridade, len(termos_encontrados),
                    aderencia_academica, aderencia_ingles, aderencia_espanhol,
                    nivel_profissional_vaga / 10
                ]]).reshape(1, -1)
                
                features_scaled = scaler.transform(features)
                probabilidade = model.predict_proba(features_scaled)[0, 1]
                score_combinado = (probabilidade * 0.3) + (match_percent * 0.4) + (similaridade * 0.2) + (aderencia_academica * 0.1)
                status, _ = calcular_status(score_combinado)
                
                resultados.append({"ID": idx + 1, "Nome": uploaded_file.name, "Score Combinado": score_combinado, "Status": status, "Probabilidade": probabilidade, "Match": match_percent})
                detalhes_candidatos.append({"ID": idx + 1, "Nome": uploaded_file.name, "Probabilidade": probabilidade, "Match": match_percent, "Aderência Acadêmica": aderencia_academica, "Aderência Inglês": aderencia_ingles, "Aderência Espanhol": aderencia_espanhol, "TermosEncontrados": ", ".join(sorted(termos_encontrados)) or "Nenhum", "TermosFaltantes": ", ".join(sorted(termos_vaga - termos_encontrados)) or "Nenhum", "TextoProcessado": cv_preprocessado_completo[:500] + "..."})
        
        if resultados:
            st.session_state.resultados_df = pd.DataFrame(resultados).sort_values("Score Combinado", ascending=False)
            # Ordena os detalhes para corresponder ao ranking
            ids_ordenados = st.session_state.resultados_df['ID'].tolist()
            st.session_state.detalhes_candidatos = sorted(detalhes_candidatos, key=lambda x: ids_ordenados.index(x['ID']))
            st.session_state.job_title = job_title
            st.rerun()
        else:
            st.warning("Nenhum candidato pôde ser processado.")

def render_metrics_page():
    """Renderiza a página com a explicação das métricas de avaliação."""
    st.title("📊 Métricas de Avaliação")
    st.markdown("""
    Esta secção detalha como cada candidato é avaliado, garantindo um processo transparente e baseado em dados.
    """)
    
    st.subheader("⚖️ Peso do Cálculo do Score Combinado")
    st.markdown("""
    O **Score Combinado** é uma média ponderada de quatro fatores principais,
    desenhado para fornecer uma visão holística da adequação de um candidato.
    - **40% Match de Competências:** A percentagem de competências técnicas requeridas na vaga que foram encontradas textualmente no currículo. É a métrica com maior peso.
    - **30% Probabilidade do Modelo:** A confiança do modelo de Machine Learning (de 0 a 100%) de que o candidato é um bom "fit", com base na análise conjunta de 7 características.
    - **20% Similaridade Textual:** Mede a semelhança de contexto e semântica entre o currículo e a descrição da vaga, usando a técnica TF-IDF.
    - **10% Aderência Académica:** Verifica se o nível de formação do candidato atende ao requisito mínimo da vaga.
    """)

    st.subheader("Como Interpretar as Métricas")
    st.markdown("""
    | Métrica             | Descrição                               | Como Interpretar                                                                    |
    |---------------------|-----------------------------------------|-------------------------------------------------------------------------------------|
    | **Score Combinado** | Avaliação final ponderada (0-100%).     | - **✅ ≥60%:** Apto<br>- **🟨 40-59%:** Em análise<br>- **❌ <40%:** Não apto        |
    | **Match** | % de competências encontradas.          | Quanto maior, mais requisitos técnicos o candidato atende.                          |
    | **Probabilidade** | Previsão do modelo (0-100%).            | A confiança do modelo de que o perfil do candidato é adequado para a vaga.          |
    | **Aderência** | Adequação aos requisitos.               | Mostra se o candidato atende aos requisitos de formação e idiomas.                  |
    """)

def render_storytelling_page():
    """Renderiza a página de Storytelling do projeto."""
    st.title("📖 Storytelling do Projeto")
    st.image("https://storage.googleapis.com/gemini-prod-us-west1-423907-8353/images/7a0402.png-ce9c0648-a0b8-45b7-8385-3dd46fef44e4", use_container_width=True)
    st.markdown("""
    ### O Problema
    No dinâmico mercado de trabalho atual, recrutadores enfrentam um desafio monumental: analisar centenas, por vezes milhares, de currículos para cada vaga. Este processo manual não é apenas demorado e repetitivo, mas também está sujeito a vieses inconscientes que podem levar à exclusão de talentos promissores. Encontrar o candidato ideal numa pilha de documentos é como procurar uma agulha num palheiro.
    ### A Solução
    Esta ferramenta nasceu para revolucionar a triagem inicial de candidatos. Utilizando o poder do **Processamento de Linguagem Natural (NLP)** e de modelos de **Machine Learning**, a nossa aplicação transforma o processo de recrutamento. Ela lê e interpreta os currículos, compara-os de forma inteligente com os requisitos da vaga e gera um **score de compatibilidade** objetivo e baseado em dados.
    ### O Impacto
    O nosso objetivo é claro: **libertar o tempo e o talento dos recrutadores**. Ao automatizar a triagem, permitimos que os profissionais de RH se foquem no que realmente importa: as interações humanas, as entrevistas estratégicas e a construção de relações com os melhores talentos. A ferramenta não substitui o recrutador, mas sim potencia as suas capacidades, oferecendo uma análise objetiva que ajuda a construir equipas mais fortes e diversificadas.
    """)

def render_tech_page():
    """Renderiza a página com as tecnologias utilizadas no projeto."""
    st.title("🛠️ Tecnologias Utilizadas")
    st.markdown("Esta aplicação foi construída com um conjunto de tecnologias modernas e robustas do ecossistema Python, focadas em ciência de dados e desenvolvimento web.")
    
    st.subheader("Interface e Visualização")
    st.markdown("- **Streamlit:** Framework principal para a criação da interface web interativa.")
    
    st.subheader("Análise e Manipulação de Dados")
    st.markdown("- **Pandas:** Utilizado para a estruturação e manipulação eficiente dos dados dos resultados.")
    st.markdown("- **NumPy:** Essencial para cálculos numéricos e a criação das 'features' para o modelo.")
    
    st.subheader("Machine Learning e NLP")
    st.markdown("- **Scikit-learn:** A biblioteca central para o nosso modelo de Machine Learning (`RandomForestClassifier`), pré-processamento (`MinMaxScaler`) e cálculo de similaridade (`TfidfVectorizer`, `cosine_similarity`).")
    st.markdown("- **NLTK (Natural Language Toolkit):** Usado para o processamento de texto, como a remoção de 'stopwords', fundamental para a análise de competências.")
    
    st.subheader("Processamento de Ficheiros")
    st.markdown("- **PyPDF2:** Biblioteca que permite a extração de texto diretamente dos ficheiros de currículo em formato PDF.")

# --- FUNÇÃO PRINCIPAL E ROTEADOR DE PÁGINAS ---

def main():
    """Função principal que define a estrutura da aplicação e a navegação."""
    st.set_page_config(page_title="Sistema de Triagem de CVs", layout="wide", initial_sidebar_state="expanded")
    
    render_sidebar()
    st.title("Sistema Inteligente de Triagem de Currículos")
    
    # Navegação por Abas no Topo da Página
    tab_main, tab_metrics, tab_story, tab_tech = st.tabs(["Análise de Currículos", "Métricas de Avaliação", "Storytelling do Projeto", "Tecnologias Utilizadas"])

    with tab_main:
        render_main_page()
    
    with tab_metrics:
        render_metrics_page()
    
    with tab_story:
        render_storytelling_page()
        
    with tab_tech:
        render_tech_page()

if __name__ == "__main__":
    main()
�