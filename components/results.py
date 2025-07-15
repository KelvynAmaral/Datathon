import streamlit as st
import pandas as pd
import plotly.express as px
from config import COLOR_MAP
from utils.ml_utils import calcular_status

def render_results(df_resultados, detalhes_candidatos, job_title):
    """Renderiza os resultados da análise em múltiplas abas."""
    st.success(f"✅ Análise concluída para {len(df_resultados)} candidatos para a vaga de **{job_title}**!")
    
    tab_dashboard, tab_ranking, tab_individual, tab_export = st.tabs([
        "🏆 Dashboard", "📊 Ranking Geral", "👤 Análise Individual", "📤 Exportar"
    ])
    
    with tab_dashboard:
        render_dashboard_tab(df_resultados)
    
    with tab_ranking:
        render_ranking_tab(df_resultados)
    
    with tab_individual:
        render_individual_tab(df_resultados, detalhes_candidatos)
    
    with tab_export:
        render_export_tab(df_resultados, detalhes_candidatos, job_title)

def render_dashboard_tab(df_resultados):
    """Renderiza a aba de Dashboard com gráfico de colunas"""
    st.header("Dashboard da Análise")
    
    # Seção 1: Tratado de candidatos
    st.subheader("Tratado de candidatos autoritários")
    col1, col2 = st.columns(2)
    with col1:
        st.write("1. [Item 1]")
    with col2:
        st.write("2. [Item 2]")
    
    # Seção 2: Métrica principal
    st.subheader("Métricas Principais")
    col1, col2, col3 = st.columns(3)
    with col1:
        score_medio = df_resultados['Score Combinado'].mean()
        st.metric("Score Médio", f"{score_medio*100:.1f}%")
    with col2:
        total_candidatos = len(df_resultados)
        st.metric("Total de Candidatos", total_candidatos)
    with col3:
        top_score = df_resultados['Score Combinado'].max()
        st.metric("Melhor Score", f"{top_score*100:.1f}%")
    
    # Seção 3: Distribuição de Status
    st.subheader("Distribuição de Status dos Candidatos")
    
    status_counts = df_resultados['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Quantidade']
    
    # Gráfico de Barras
    fig = px.bar(
        status_counts,
        x='Status',
        y='Quantidade',
        color='Status',
        color_discrete_map=COLOR_MAP,
        text='Quantidade',
        labels={'Quantidade': 'Número de Candidatos', 'Status': ''}
    )
    
    # Personalização do gráfico
    fig.update_traces(
        textposition='outside',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        textfont_size=12
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title='Número de Candidatos',
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        xaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar os dados em formato de tabela
    with st.expander("Ver dados detalhados"):
        st.dataframe(status_counts, hide_index=True)

def render_ranking_tab(df_resultados):
    """Renderiza a aba de Ranking com filtro de Top 10"""
    st.header("Ranking de Candidatos")
    
    # Filtros
    col1, col2 = st.columns([3, 1])
    with col1:
        status_options = st.multiselect(
            "Filtrar por Status:",
            options=df_resultados['Status'].unique(),
            default=df_resultados['Status'].unique()
        )
    
    with col2:
        top_10 = st.checkbox("Mostrar apenas Top 10", value=False)
    
    # Aplicar filtros
    df_filtrado = df_resultados[df_resultados['Status'].isin(status_options)]
    
    if top_10:
        df_filtrado = df_filtrado.nlargest(10, 'Score Combinado')
    
    # Mostrar dataframe
    st.dataframe(
        df_filtrado.sort_values('Score Combinado', ascending=False),
        column_config={
            "Score Combinado": st.column_config.ProgressColumn(
                "Score",
                format="%.1f%%",
                min_value=0,
                max_value=1
            ),
            "Probabilidade": st.column_config.NumberColumn(
                "Prob. ML",
                format="%.1f%%"
            ),
            "Match": st.column_config.NumberColumn(
                "Match",
                format="%.1f%%"
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Mostrar estatísticas rápidas quando filtro Top 10 estiver ativo
    if top_10 and len(df_filtrado) > 0:
        avg_score = df_filtrado['Score Combinado'].mean()
        min_score_top10 = df_filtrado['Score Combinado'].min()
        max_score_top10 = df_filtrado['Score Combinado'].max()
        
        st.markdown(f"""
        **Estatísticas do Top 10:**
        - 📊 Score médio: **{avg_score*100:.1f}%**
        - 🏆 Melhor score: **{max_score_top10*100:.1f}%**
        - 🔍 Pior score do Top 10: **{min_score_top10*100:.1f}%**
        """)

def render_individual_tab(df_resultados, detalhes_candidatos):
    """Renderiza a aba de Análise Individual"""
    st.header("Análise Detalhada por Candidato")
    
    candidato_id = st.selectbox(
        "Selecione o Candidato:",
        options=df_resultados['ID'].tolist(),
        format_func=lambda x: f"{x} - {df_resultados[df_resultados['ID'] == x]['Nome'].iloc[0]}"
    )
    
    candidato = next((c for c in detalhes_candidatos if c['ID'] == candidato_id), None)
    if not candidato:
        st.warning("Candidato não encontrado.")
        return
    
    score = df_resultados[df_resultados['ID'] == candidato_id]['Score Combinado'].iloc[0]
    status, _ = calcular_status(score)
    
    with st.container(border=True):
        st.subheader(f"📄 {candidato['Nome']}")
        st.markdown(f"**Status:** {status}")
        
        cols = st.columns(3)
        cols[0].metric("Score Total", f"{score:.1%}")
        cols[1].metric("Match Técnico", f"{candidato['Match']:.1%}")
        cols[2].metric("Probabilidade ML", f"{candidato['Probabilidade']:.1%}")
        
        st.divider()
        
        cols = st.columns(3)
        cols[0].progress(candidato['Aderência Acadêmica'], text="Aderência Acadêmica")
        cols[1].progress(candidato['Aderência Inglês'], text="Aderência Inglês")
        cols[2].progress(candidato['Aderência Espanhol'], text="Aderência Espanhol")
        
        st.divider()
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**✅ Competências Encontradas:**")
            st.write(candidato['TermosEncontrados'] or "Nenhuma competência encontrada.")
        
        with cols[1]:
            st.markdown("**⚠️ Competências Faltantes:**")
            st.write(candidato['TermosFaltantes'] or "Todas competências atendidas.")
        
        st.markdown("**📝 Texto Processado:**")
        st.text_area(
            "Texto Analisado",
            value=candidato['TextoProcessado'],
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )

def render_export_tab(df_resultados, detalhes_candidatos, job_title):
    """Renderiza a aba de Exportação"""
    st.header("Exportar Resultados")
    
    df_detalhes = pd.DataFrame(detalhes_candidatos)
    df_completo = pd.merge(
        df_resultados,
        df_detalhes,
        on=['ID', 'Nome', 'Probabilidade', 'Match'],
        how='left'
    )
    
    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            "📊 Exportar Resumo (CSV)",
            data=df_resultados.to_csv(index=False, sep=';', encoding='utf-8-sig'),
            file_name=f"resumo_candidatos_{job_title.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with cols[1]:
        st.download_button(
            "📝 Exportar Detalhes (CSV)",
            data=df_completo.to_csv(index=False, sep=';', encoding='utf-8-sig'),
            file_name=f"detalhes_candidatos_{job_title.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    st.divider()
    st.subheader("Pré-visualização dos Dados")
    
    tab1, tab2 = st.tabs(["Resumo", "Detalhes"])
    with tab1:
        st.dataframe(df_resultados, hide_index=True)
    
    with tab2:
        st.dataframe(df_completo, hide_index=True)