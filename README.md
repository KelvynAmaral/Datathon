# Orientação para criar um venv e uma imagem no docker
- Criar um ambiente virtual:
  
    python -m venv venv

- Ativar o ambiente: 

   venv\Scripts\activate

- Construção da Imagem Docker

   docker build . -t streamlit_image  ou docker build -t streamlit_image .

- Listar images

  docker images
  
- Executar um Contêiner Docker:
    docker run --name streamlit_container -p 8501:8501 -d -v ${pwd}:/code streamlit_image
