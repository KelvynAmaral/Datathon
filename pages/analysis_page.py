import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from config import (
    MAPA_NIVEL_PROFISSIONAL,
    MAPA_ACADEMICO,
    MAPA_IDIOMA
)
from utils.file_utils import extract_text_from_pdf, load_models
from utils.text_processing import (
    preprocessar_texto,
    extrair_competencias,
    mapear_nivel,
    calcular_similaridade_texto,
    setup_nltk
)
from utils.ml_utils import calcular_status, calcular_score_combinado
from components.results import render_results

def render_main_page():
    """Renderiza a página principal de análise"""
    st.header("🎯 Ferramenta de Análise de Currículos")
    
    with st.expander("ℹ️ Como usar esta ferramenta", expanded=False):
        st.markdown("""
        1. Preencha os detalhes da vaga
        2. Adicione os currículos em PDF
        3. Clique em "Analisar Candidatos"
        4. Explore os resultados nas abas
        """)
    
    with st.container(border=True):
        with st.form("vaga_form"):
            job_title = st.text_input("Título da Vaga*", placeholder="Ex: Engenheiro de Dados Sênior")
            job_requirements = st.text_area("Competências Requeridas*", placeholder="Ex: Python, SQL...", height=150)
            
            col1, col2 = st.columns(2)
            with col1:
                job_academic_level = st.selectbox("Nível Acadêmico*", options=list(MAPA_ACADEMICO.keys()))
                job_english = st.selectbox("Inglês*", options=list(MAPA_IDIOMA.keys()))
            with col2:
                job_professional_level = st.selectbox("Nível Profissional*", options=list(MAPA_NIVEL_PROFISSIONAL.keys()))
                job_spanish = st.selectbox("Espanhol", options=list(MAPA_IDIOMA.keys()))
            
            uploaded_files = st.file_uploader("Currículos (PDF)*", type=["pdf"], accept_multiple_files=True)
            
            submitted = st.form_submit_button("🚀 Analisar Candidatos", type="primary")

    if submitted:
        process_submission(job_title, job_requirements, uploaded_files, 
                         job_academic_level, job_english, job_spanish, job_professional_level)
    elif "resultados_df" in st.session_state:
        render_results(
            st.session_state.resultados_df,
            st.session_state.detalhes_candidatos,
            st.session_state.job_title
        )

def process_submission(job_title, job_requirements, uploaded_files, 
                      job_academic_level, job_english, job_spanish, job_professional_level):
    """Processa os currículos submetidos"""
    if not all([job_title, job_requirements, uploaded_files]):
        st.error("Preencha todos os campos obrigatórios (*)")
        return

    with st.spinner("Processando currículos..."):
        try:
            model, scaler, vectorizer = load_models()
            stopwords_pt = setup_nltk()
            
            # Processamento inicial
            termos_vaga = extrair_competencias(job_requirements)
            req_preprocessados = preprocessar_texto(" ".join(termos_vaga), stopwords_pt)
            
            # Configuração de níveis
            nivel_academico_vaga = MAPA_ACADEMICO[job_academic_level.lower()]
            nivel_ingles_vaga = MAPA_IDIOMA[job_english.lower()]
            nivel_profissional_vaga = MAPA_NIVEL_PROFISSIONAL[job_professional_level.lower()]
            
            resultados = []
            detalhes_candidatos = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                cv_text_raw = extract_text_from_pdf(BytesIO(uploaded_file.getvalue()))
                if not cv_text_raw:
                    continue
                
                # Processamento do currículo
                cv_preprocessado = preprocessar_texto(cv_text_raw, stopwords_pt)
                termos_encontrados = {term for term in termos_vaga if term in cv_text_raw.lower()}
                match_percent = len(termos_encontrados) / len(termos_vaga) if termos_vaga else 0
                
                # Cálculo de similaridade
                similaridade = calcular_similaridade_texto(req_preprocessados, cv_preprocessado, vectorizer)
                
                # Cálculo de scores
                features = np.array([[
                    match_percent,
                    similaridade,
                    len(termos_encontrados),
                    nivel_academico_vaga,
                    MAPA_IDIOMA[job_english.lower()],
                    MAPA_IDIOMA[job_spanish.lower()],
                    nivel_profissional_vaga / 10
                ]])
                
                features_scaled = scaler.transform(features)
                probabilidade = model.predict_proba(features_scaled)[0, 1]
                score = calcular_score_combinado(probabilidade, match_percent, similaridade, nivel_academico_vaga)
                status, _ = calcular_status(score)
                
                # Armazena resultados
                resultados.append({
                    "ID": idx + 1,
                    "Nome": uploaded_file.name,
                    "Score Combinado": score,
                    "Status": status,
                    "Probabilidade": probabilidade,
                    "Match": match_percent
                })
                
                # Dicionário com todas as informações do candidato (incluindo as novas aderências)
                detalhes_candidatos.append({
                    "ID": idx + 1,
                    "Nome": uploaded_file.name,
                    "Probabilidade": probabilidade,
                    "Match": match_percent,
                    "TermosEncontrados": ", ".join(sorted(termos_encontrados)) or "Nenhum",
                    "TermosFaltantes": ", ".join(sorted(termos_vaga - termos_encontrados)) or "Nenhum",
                    "TextoProcessado": cv_preprocessado[:1000] + "...",
                    "Aderência Acadêmica": nivel_academico_vaga / 10,
                    "Aderência Inglês": MAPA_IDIOMA[job_english.lower()] / 10,
                    "Aderência Espanhol": MAPA_IDIOMA[job_spanish.lower()] / 10
                })
            
            if resultados:
                st.session_state.resultados_df = pd.DataFrame(resultados)
                st.session_state.detalhes_candidatos = detalhes_candidatos
                st.session_state.job_title = job_title
                st.rerun()
                
        except Exception as e:
            st.error(f"Erro no processamento: {str(e)}")