# app.py - Sistema de Recomendação de Talentos por Vaga

import streamlit as st
import pandas as pd
import json
import gdown
import re
import unicodedata
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(page_title="Sistema de Recomendação de Talentos", layout="wide")
st.title("Dashboard de Matching entre Vagas e Candidatos")
st.subheader("Selecione uma vaga na aba lateral para visualizar os candidatos mais compatíveis")

# Carregamento dos dados JSON (apenas se ainda não estiverem na sessão)
if 'df_Vagas' not in st.session_state or 'df_Applicants' not in st.session_state:
    file_Applicants = '17859ae_Ki5CImI9-1lhJ335GMDW0f2Qr'
    file_Vagas      = '1YKM7yDTzjHJVf82l2RxEx-SuLxFxCxrl'

    # Baixa os arquivos do Google Drive
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_Applicants}', 'applicants.json', quiet=False)
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_Vagas}', 'vagas.json', quiet=False)

    # Leitura dos arquivos com encoding utf-8
    with open('applicants.json', 'r', encoding='utf-8') as applicants_file:
        data_Applicants = json.load(applicants_file)

    with open('vagas.json', 'r', encoding='utf-8') as vagas_file:
        data_Vagas = json.load(vagas_file)

    # Transformar applicants em DataFrame
    records = []
    for prof_id, profile_info in data_Applicants.items():
        record = {"id_applicant": prof_id}
        for section_name, section_data in profile_info.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    record[f"{section_name}__{key}"] = value
            else:
                record[section_name] = section_data
        records.append(record)
    st.session_state.df_Applicants = pd.DataFrame(records)

    # Transformar vagas em DataFrame
    records = []
    for prof_id, profile_info in data_Vagas.items():
        record = {"id_vaga": prof_id}
        for section_name, section_data in profile_info.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    record[f"{section_name}__{key}"] = value
            else:
                record[section_name] = section_data
        records.append(record)
    st.session_state.df_Vagas = pd.DataFrame(records)

    # Limpeza de memória
    del data_Vagas
    del data_Applicants
    gc.collect()

# 💾 Acesso aos DataFrames da sessão
df_Vagas = st.session_state.df_Vagas
df_Applicants = st.session_state.df_Applicants

# Pré-processamento inicial
df_Vagas['informacoes_basicas__data_requicisao'] = pd.to_datetime(
    df_Vagas['informacoes_basicas__data_requicisao'], format='%d-%m-%Y', errors='coerce'
)
df_Vagas['mes_ano'] = df_Vagas['informacoes_basicas__data_requicisao'].dt.strftime('%b.%Y')

# Filtros
st.sidebar.header("Filtros")
meses_disponiveis = sorted(df_Vagas['mes_ano'].dropna().unique(), key=lambda x: pd.to_datetime(x, format='%b.%Y'))
mes_selecionado = st.sidebar.selectbox("Selecione o mês:", meses_disponiveis)

df_filtrado = df_Vagas[df_Vagas['mes_ano'] == mes_selecionado]
vagas_disponiveis = sorted(df_filtrado['informacoes_basicas__titulo_vaga'].dropna().unique())
vaga_selecionada = st.sidebar.selectbox("Título da vaga:", vagas_disponiveis)

# 🔍 Matching de candidatos 
vaga_info = df_filtrado[df_filtrado['informacoes_basicas__titulo_vaga'] == vaga_selecionada].iloc[0]
descricao_vaga = vaga_info.get('perfil_vaga__competencia_tecnicas_e_comportamentais', '')

def clean_text(text):
    text = re.sub(r'\n', ' ', str(text))
    text = re.sub(r'[^a-zA-Z0-9\u00C0-\u00FF ]', '', text)
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

descricao_limpa = clean_text(descricao_vaga)
cvs_limpos = df_Applicants['cv_pt'].fillna('').apply(clean_text).tolist()

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform([descricao_limpa] + cvs_limpos)
scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

st.write("Colunas disponíveis:", df_Applicants.columns.tolist())

df_Applicants['match_score'] = scores
top_candidatos = df_Applicants.sort_values(by='match_score', ascending=False).head(10)

# 📊 Exibição final
st.markdown(f"### Top 10 Candidatos para a vaga: {vaga_selecionada}")
st.dataframe(top_candidatos[['nome', 'match_score', 'cv_pt']], use_container_width=True)
