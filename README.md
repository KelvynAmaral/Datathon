# 🎯 Sistema Inteligente de Triagem de Currículos

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff69b4.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)
![Status](https://img.shields.io/badge/status-concluído-success)

## 📖 Sobre o Projeto (Storytelling)

No dinâmico mercado de trabalho atual, recrutadores enfrentam um desafio monumental: analisar centenas, por vezes milhares, de currículos para cada vaga. Este processo manual não é apenas demorado e repetitivo, mas também está sujeito a vieses inconscientes que podem levar à exclusão de talentos promissores. Encontrar o candidato ideal numa pilha de documentos é como procurar uma agulha num palheiro.

Esta ferramenta nasceu para revolucionar a triagem inicial de candidatos. Utilizando o poder do **Processamento de Linguagem Natural (NLP)** e de modelos de **Machine Learning**, a nossa aplicação transforma o processo de recrutamento. Ela lê e interpreta os currículos, compara-os de forma inteligente com os requisitos da vaga e gera um **score de compatibilidade** objetivo e baseado em dados.

O nosso objetivo é claro: **libertar o tempo e o talento dos recrutadores**. Ao automatizar a triagem, permitimos que os profissionais de RH se foquem no que realmente importa: as interações humanas, as entrevistas estratégicas e a construção de relações com os melhores talentos.

---

## ✨ Funcionalidades Principais

* **Análise de Múltiplos Currículos:** Faça o upload de vários currículos em formato PDF de uma só vez.
* **Score de Compatibilidade Ponderado:** Cada candidato recebe um score final baseado em 4 fatores:
    * **40% Match de Competências:** Análise da presença de palavras-chave.
    * **30% Probabilidade do Modelo:** Predição de um modelo `RandomForestClassifier` treinado.
    * **20% Similaridade Textual:** Análise de contexto com TF-IDF e Similaridade de Cossenos.
    * **10% Aderência Académica:** Comparação do nível de formação.
* **Dashboard Interativo:** Visualize um resumo da análise com gráficos e métricas principais.
* **Ranking e Filtros:** Classifique os candidatos pelo score e filtre-os por status para uma análise focada.
* **Análise Individual Detalhada:** Explore um "card" completo para cada candidato com todas as métricas, competências encontradas e em falta.
* **Exportação de Resultados:** Faça o download dos resultados em formato CSV para relatórios e análises offline.

---

## 📸 Demonstração da Aplicação


*Interface principal da ferramenta de análise de currículos.*

---

## 🛠️ Tecnologias Utilizadas

A aplicação foi construída com um conjunto de tecnologias modernas e robustas do ecossistema Python:

* **Interface e Visualização:**
    * **Streamlit:** Framework principal para a criação da interface web interativa.
    * **Plotly Express:** Para a criação de gráficos dinâmicos no dashboard.

* **Análise e Manipulação de Dados:**
    * **Pandas:** Para a estruturação e manipulação eficiente dos dados.
    * **NumPy:** Essencial para cálculos numéricos e a criação das 'features'.

* **Machine Learning e NLP:**
    * **Scikit-learn:** A biblioteca central para o modelo de Machine Learning, pré-processamento e cálculo de similaridade.
    * **NLTK (Natural Language Toolkit):** Usado para o processamento de texto (remoção de 'stopwords').

* **Processamento de Ficheiros:**
    * **PyPDF2:** Para a extração de texto diretamente dos ficheiros de currículo em PDF.

---

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e executar a aplicação no seu ambiente local.

### Pré-requisitos

* Python 3.9 ou superior
* `pip` (gestor de pacotes do Python)

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Data-Analitycs-Pos-Tech-Fiap/Datathon.git
    ```

2.  **Crie e ative um ambiente virtual** (altamente recomendado):
    ```bash
    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Para macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Este comando irá instalar todas as bibliotecas necessárias, como Streamlit, Pandas, Scikit-learn, etc.*

### Execução

1.  Com o seu ambiente virtual ativado, execute o seguinte comando no terminal:
    ```bash
    streamlit run app.py
    ```

2.  A aplicação será aberta automaticamente no seu navegador web.

---

## 👨‍💻 Autores

Este projeto foi desenvolvido por:

* Kelvyn Candido
* Anderson Silva
* Sandra
* Michael
* Evandro Gardin
