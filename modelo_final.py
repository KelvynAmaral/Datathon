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

# Configurações iniciais
nltk.download('stopwords')
stopwords = stopwords.words('portuguese')

# Carregar modelo, scaler e vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load('modelo_rf_final.pkl')
        scaler = joblib.load('scaler_final.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, scaler, vectorizer
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        st.stop()

model, scaler, tfidf_vectorizer = load_models()

# Funções auxiliares
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Erro ao ler PDF: {str(e)}")
        return ""

def preprocessar_texto(texto):
    """Pré-processamento rigoroso para similaridade"""
    if not texto:
        return ""
    
    texto = str(texto).lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    palavras = [palavra for palavra in texto.split()
                if palavra not in stopwords and len(palavra) >= 3]
    
    return ' '.join(palavras)

def calcular_relevancia_termos(termos_encontrados, total_termos_vaga):
    """Calcula a relevância baseada na porcentagem de termos encontrados"""
    if not termos_encontrados or total_termos_vaga == 0:
        return 0
    return min(len(termos_encontrados) / total_termos_vaga, 1.0)

def mapear_nivel_academico(nivel):
    if not nivel or pd.isna(nivel) or str(nivel).strip() == '':
        return 0
    
    nivel = str(nivel).lower()
    if "doutorado" in nivel: return 5
    if "mestrado" in nivel: return 4
    if "pós-graduação" in nivel or "pós graduação" in nivel: return 3
    if "ensino superior" in nivel: return 2
    if "ensino técnico" in nivel: return 1.5
    if "ensino médio" in nivel: return 1
    if "ensino fundamental" in nivel: return 0.5
    return 0

def mapear_nivel_idioma(nivel):
    if not nivel or pd.isna(nivel) or str(nivel).strip() == '':
        return 0
    
    nivel = str(nivel).lower()
    if "nenhum" in nivel: return 0
    if "básico" in nivel: return 1
    if "intermediário" in nivel: return 2
    if "avançado" in nivel: return 3
    if "fluente" in nivel: return 4
    return 0

mapa_nivel_profissional = {
    'aprendiz': 1, 'trainee': 2, 'auxiliar': 3, 'assistente': 4,
    'técnico de nível médio': 5, 'júnior': 5.5, 'analista': 6,
    'pleno': 7, 'supervisor': 7, 'líder': 7.5, 'sênior': 8,
    'especialista': 9, 'coordenador': 9, 'gerente': 10
}

def get_nivel_profissional(nivel):
    if not nivel or pd.isna(nivel) or str(nivel).strip() == '':
        return 0
    return mapa_nivel_profissional.get(str(nivel).lower(), 0)

def calcular_status(score_combinado):
    """Calcula o status baseado no score combinado"""
    if score_combinado >= 0.6:
        return "✅ Apto (Alto)", "green"
    elif score_combinado >= 0.5:
        return "🟨 Em análise (Médio)", "orange"
    else:
        return "❌ Não apto (Baixo)", "red"

def show_metrics_explanation():
    """Exibe as explicações para o recrutador"""
    with st.expander("🔍 Como Interpretar as Métricas", expanded=True):
        st.markdown("""
        ### 📊 Métricas de Avaliação
        
        | Métrica | Descrição | Como Interpretar |
        |---------|-----------|------------------|
        | **Score Combinado** | Avaliação final (0-100%) | - ✅ ≥60%: Apto<br>- 🟨 40-59%: Em análise<br>- ❌ <40%: Não apto |
        | **Match** | % competências encontradas | Quanto maior, mais requisitos atendidos |
        | **Probabilidade** | Previsão do modelo (0-100%) | Confiança na adequação técnica |
        | **Aderência** | Adequação aos requisitos | ✅=Atende, △=Parcial, ❌=Não atende |
        
        ### ⚖️ Peso do Cálculo
        - **40% Match de Competências** (termos da vaga no currículo)
        - **30% Probabilidade do Modelo** (análise de 7 fatores técnicos)
        - **20% Similaridade Textual** (análise semântica)
        - **10% Aderência Acadêmica**
        """)

def main():
    st.set_page_config(page_title="Sistema Avançado de Classificação de Candidatos", layout="wide")
    st.title("📊 Sistema Avançado de Classificação de Candidatos")
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Sobre o Sistema")
        st.markdown("""
        **Critérios de Avaliação:**
        - ✅ **Apto**: Score ≥ 60%
        - 🟨 **Em análise**: 40% ≤ Score < 60%
        - ❌ **Não apto**: Score < 40%
        
        **Score Combinado:**
        - 30% Probabilidade do modelo
        - 40% Match de competências
        - 20% Similaridade textual
        - 10% Aderência acadêmica
        """)
        st.divider()
        st.markdown(f"🔄 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    # Seção de explicações
    show_metrics_explanation()

    # Seção de informações da vaga
    with st.form("vaga_form"):
        st.header("📝 Informações da Vaga")
        job_title = st.text_input("Título da Vaga*", placeholder="Ex: Desenvolvedor Python Pleno")
        job_requirements = st.text_area("Competências Técnicas Requeridas*",
                                          placeholder="Liste as habilidades-chave separadas por vírgula",
                                          height=150)
        
        col1, col2 = st.columns(2)
        with col1:
            job_academic_level = st.selectbox(
                "Nível Acadêmico Requerido*",
                ["", "Ensino Fundamental", "Ensino Médio", "Ensino Técnico",
                 "Ensino Superior", "Pós-Graduação", "Mestrado", "Doutorado"]
            )
            job_english = st.selectbox(
                "Inglês Exigido*",
                ["", "Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"],
                index=0
            )
        with col2:
            job_professional_level = st.selectbox(
                "Nível Profissional*",
                ["", "Aprendiz", "Trainee", "Auxiliar", "Assistente", "Técnico de Nível Médio",
                 "Júnior", "Analista", "Pleno", "Supervisor", "Líder", "Sênior",
                 "Especialista", "Coordenador", "Gerente"]
            )
            job_spanish = st.selectbox(
                "Espanhol Exigido",
                ["", "Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"],
                index=0
            )
        
        st.header("👤 Informações dos Candidatos")
        uploaded_files = st.file_uploader(
            "Selecione os currículos em PDF*",
            type=["pdf"],
            accept_multiple_files=True,
            help="Selecione vários arquivos de uma vez"
        )
        
        # Seção para cada candidato preencher seus dados
        st.subheader("Dados dos Candidatos")
        candidates_data = []
        
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"Candidato {i+1}: {uploaded_file.name}"):
                    cols = st.columns(3)
                    with cols[0]:
                        academic_level = st.selectbox(
                            f"Nível Acadêmico*",
                            ["", "Ensino Fundamental", "Ensino Médio", "Ensino Técnico",
                             "Ensino Superior", "Pós-Graduação", "Mestrado", "Doutorado"],
                            key=f"academic_{i}"
                        )
                    with cols[1]:
                        english_level = st.selectbox(
                            f"Inglês*",
                            ["", "Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"],
                            key=f"english_{i}"
                        )
                    with cols[2]:
                        spanish_level = st.selectbox(
                            f"Espanhol",
                            ["", "Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"],
                            key=f"spanish_{i}"
                        )
                    
                    candidates_data.append({
                        "file": uploaded_file,
                        "academic_level": academic_level,
                        "english_level": english_level,
                        "spanish_level": spanish_level
                    })
        
        submitted = st.form_submit_button("🔍 Avaliar Candidatos")
        st.markdown("*Campos obrigatórios")

    if submitted:
        if not uploaded_files:
            st.error("🚨 Por favor, envie pelo menos um currículo")
            st.stop()
        
        campos_obrigatorios = {
            "Título da Vaga": job_title,
            "Competências Requeridas": job_requirements,
            "Nível Acadêmico da Vaga": job_academic_level,
            "Nível Profissional da Vaga": job_professional_level,
            "Inglês Exigido": job_english
        }
        
        campos_faltantes = [nome for nome, valor in campos_obrigatorios.items() if not valor or str(valor).strip() == '']
        
        if campos_faltantes:
            st.error(f"🚨 Por favor, preencha todos os campos obrigatórios: {', '.join(campos_faltantes)}")
            st.stop()

        # Verificar dados dos candidatos
        for i, candidate in enumerate(candidates_data):
            if not candidate['academic_level'] or not candidate['english_level']:
                st.error(f"🚨 Por favor, preencha os dados obrigatórios para o Candidato {i+1}")
                st.stop()

        resultados = []
        detalhes_candidatos = []
        
        # Pré-processamento dos requisitos da vaga
        req_preprocessados = preprocessar_texto(job_requirements)
        termos_vaga = set(req_preprocessados.split())
        total_termos_vaga = len(termos_vaga)
        
        # Mapear níveis da vaga
        nivel_academico_vaga = mapear_nivel_academico(job_academic_level)
        nivel_profissional_vaga = get_nivel_profissional(job_professional_level)
        nivel_profissional_norm = nivel_profissional_vaga / 10
        nivel_ingles_vaga = mapear_nivel_idioma(job_english)
        nivel_espanhol_vaga = mapear_nivel_idioma(job_spanish)
        
        # Processar cada candidato
        for idx, candidate in enumerate(candidates_data):
            try:
                with st.spinner(f"Processando candidato {idx+1}/{len(candidates_data)}..."):
                    # Extrair texto do PDF
                    candidate_cv = extract_text_from_pdf(candidate['file'])
                    
                    if not candidate_cv:
                        st.warning(f"Arquivo {candidate['file'].name} vazio ou não processado")
                        continue
                    
                    # Pré-processamento
                    cv_preprocessados = preprocessar_texto(candidate_cv)
                    
                    # Cálculo de termos
                    termos_cv = set(cv_preprocessados.split())
                    termos_encontrados = list(termos_vaga & termos_cv)
                    qtd_termos_encontrados = len(termos_encontrados)
                    match_percent = calcular_relevancia_termos(termos_encontrados, total_termos_vaga)
                    
                    # Similaridade
                    try:
                        corpus = [req_preprocessados, cv_preprocessados]
                        tfidf_matrix = tfidf_vectorizer.transform(corpus)
                        similaridade = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        similaridade = (similaridade + 1) / 2  # Normalizar para 0-1
                    except Exception:
                        similaridade = 0
                    
                    # Mapear níveis do candidato
                    nivel_academico_candidato = mapear_nivel_academico(candidate['academic_level'])
                    nivel_ingles_candidato = mapear_nivel_idioma(candidate['english_level'])
                    nivel_espanhol_candidato = mapear_nivel_idioma(candidate['spanish_level'] or "Nenhum")
                    
                    # Calcular aderência
                    aderencia_academica = 1 if nivel_academico_candidato >= nivel_academico_vaga else 0.5 if nivel_academico_candidato >= nivel_academico_vaga - 1 else 0
                    aderencia_ingles = 1 if nivel_ingles_candidato >= nivel_ingles_vaga else 0.5 if nivel_ingles_candidato >= nivel_ingles_vaga - 1 else 0
                    aderencia_espanhol = 1 if nivel_espanhol_candidato >= nivel_espanhol_vaga else 0.5 if nivel_espanhol_candidato >= nivel_espanhol_vaga - 1 else 0
                    
                    # Features para o modelo
                    features = np.array([
                        match_percent,
                        similaridade,
                        qtd_termos_encontrados,
                        aderencia_academica,
                        aderencia_ingles,
                        aderencia_espanhol,
                        nivel_profissional_norm
                    ]).reshape(1, -1)
                    
                    # Normalização e predição
                    try:
                        features_scaled = scaler.transform(features)
                        probabilidade = model.predict_proba(features_scaled)[0, 1]
                    except Exception:
                        probabilidade = 0.5
                    
                    # Calcular score combinado
                    score_combinado = (probabilidade * 0.3) + (match_percent * 0.4) + (similaridade * 0.2) + (aderencia_academica * 0.1)
                    status, cor_status = calcular_status(score_combinado)
                    
                    # Resultados para exibição
                    resultados.append({
                        "ID": idx+1,
                        "Nome": candidate['file'].name,
                        "Probabilidade": probabilidade,
                        "Match": match_percent,
                        "Status": status,
                        "Cor Status": cor_status,
                        "Termos": f"{qtd_termos_encontrados}/{total_termos_vaga}",
                        "Similaridade": similaridade,
                        "Nível Acadêmico": candidate['academic_level'],
                        "Inglês": candidate['english_level'],
                        "Espanhol": candidate['spanish_level'] or "Nenhum",
                        "Score Combinado": score_combinado
                    })
                    
                    detalhes_candidatos.append({
                        "ID": idx+1,
                        "Nome": candidate['file'].name,
                        "TermosEncontrados": ", ".join(termos_encontrados) or "Nenhum",
                        "TermosFaltantes": ", ".join(sorted(termos_vaga - termos_cv)[:10]) or "Nenhum",
                        "TextoProcessado": cv_preprocessados[:500] + "...",
                        "Probabilidade": probabilidade,
                        "Match": match_percent,
                        "Aderência Acadêmica": aderencia_academica,
                        "Aderência Inglês": aderencia_ingles,
                        "Aderência Espanhol": aderencia_espanhol
                    })

            except Exception as e:
                st.error(f"Erro ao processar {candidate['file'].name}: {str(e)}")
                continue

        # Exibir resultados
        st.success(f"✅ Análise concluída para {len(resultados)} candidatos!")
        
        # Tabela resumo
        st.header("📊 Resultados Consolidados")
        df_resultados = pd.DataFrame(resultados).sort_values("Score Combinado", ascending=False)
        
        st.dataframe(
            df_resultados[["ID", "Nome", "Score Combinado", "Status", "Probabilidade", "Match", "Termos"]],
            column_config={
                "Score Combinado": st.column_config.ProgressColumn(
                    "Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                ),
                "Probabilidade": st.column_config.ProgressColumn(
                    "Probabilidade",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                ),
                "Match": st.column_config.ProgressColumn(
                    "Match",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                )
            },
            hide_index=True,
            use_container_width=True
        )

        # Top candidatos
        st.subheader("🏆 Melhores Candidatos")
        top_candidatos = df_resultados.head(3)
        
        cols = st.columns(3)
        for idx, (_, candidato) in enumerate(top_candidatos.iterrows()):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**{candidato['Nome']}**")
                    
                    # Linha com Probabilidade e Match
                    col_metrics = st.columns(2)
                    with col_metrics[0]:
                        st.metric("Probabilidade", f"{candidato['Probabilidade']:.1%}")
                    with col_metrics[1]:
                        st.metric("Match", f"{candidato['Match']:.1%}")
                    
                    st.markdown(f"**Status:** :{candidato['Cor Status'].lower()}[{candidato['Status']}]")
                    st.markdown(f"**Score Combinado:** {candidato['Score Combinado']:.1%}")
                    st.markdown(f"**Nível Acadêmico:** {candidato['Nível Acadêmico']}")
                    st.markdown(f"**Inglês:** {candidato['Inglês']}")
                    st.markdown(f"**Espanhol:** {candidato['Espanhol']}")
                    
                    detalhes = next((d for d in detalhes_candidatos if d["ID"] == candidato["ID"]), None)
                    if detalhes:
                        with st.expander("Detalhes Técnicos"):
                            st.markdown("**Competências encontradas:**")
                            st.info(detalhes["TermosEncontrados"])
                            st.markdown("**Principais faltantes:**")
                            st.warning(detalhes["TermosFaltantes"])
                            st.markdown("**Aderência:**")
                            col_ader = st.columns(3)
                            with col_ader[0]: st.metric("Acadêmica", f"{detalhes['Aderência Acadêmica']*100:.0f}%")
                            with col_ader[1]: st.metric("Inglês", f"{detalhes['Aderência Inglês']*100:.0f}%")
                            with col_ader[2]: st.metric("Espanhol", f"{detalhes['Aderência Espanhol']*100:.0f}%")

        # Detalhes completos
        st.subheader("🔍 Detalhes por Candidato")
        for detalhe in detalhes_candidatos:
            with st.expander(f"Candidato {detalhe['ID']} - {detalhe['Nome']} (Score: {detalhe['Probabilidade']*0.3 + detalhe['Match']*0.4 + detalhe['Aderência Acadêmica']*0.1:.1%})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Competências encontradas:**")
                    st.info(detalhe["TermosEncontrados"])
                with col2:
                    st.markdown("**Principais competências faltantes:**")
                    st.warning(detalhe["TermosFaltantes"])
                
                st.markdown("**Métricas Principais:**")
                cols_metrics = st.columns(3)
                with cols_metrics[0]:
                    st.metric("Probabilidade", f"{detalhe['Probabilidade']:.1%}")
                with cols_metrics[1]:
                    st.metric("Match", f"{detalhe['Match']:.1%}")
                with cols_metrics[2]:
                    score = (detalhe['Probabilidade']*0.3 + detalhe['Match']*0.4 + detalhe['Aderência Acadêmica']*0.1)
                    st.metric("Score Combinado", f"{score:.1%}")
                
                st.markdown("**Aderência aos Requisitos:**")
                cols_ader = st.columns(3)
                with cols_ader[0]:
                    st.metric("Acadêmica", f"{detalhe['Aderência Acadêmica']*100:.0f}%",
                              help="1.0 = Atende completamente, 0.5 = Parcialmente, 0 = Não atende")
                with cols_ader[1]:
                    st.metric("Inglês", f"{detalhe['Aderência Inglês']*100:.0f}%",
                              help="Comparação com o nível exigido pela vaga")
                with cols_ader[2]:
                    st.metric("Espanhol", f"{detalhe['Aderência Espanhol']*100:.0f}%",
                              help="Comparação com o nível exigido pela vaga")
                
                st.markdown("**Trecho processado do currículo:**")
                st.text(detalhe["TextoProcessado"])

        # Exportar resultados
        st.divider()
        with st.container(border=True):
            st.subheader("📤 Exportar Resultados")
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_resumo = df_resultados.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
                st.download_button(
                    label="📊 Exportar Resumo CSV",
                    data=csv_resumo,
                    file_name="resumo_candidatos.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp2:
                df_detalhes_exp = pd.DataFrame(detalhes_candidatos)
                csv_detalhes = df_detalhes_exp.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
                st.download_button(
                    label="📝 Exportar Detalhes CSV",
                    data=csv_detalhes,
                    file_name="detalhes_candidatos.csv",
                    mime="text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()