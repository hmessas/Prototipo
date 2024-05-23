import streamlit as st
import time
import yfinance as yf

@st.cache_data
def download():
    df= yf.download('AAPL')
    return df
# Título do protótipo
st.title("Predict AI (Protótipo)")

# Tabs
tabs = ["Planos", "Cliente Normal", "Cliente Premium", "Documentação"]
tab = st.tabs(tabs)

# Função para exibir texto explicativo
def show_explanation(text):
    st.write(text)

# Função para esconder a sidebar
def hide_sidebar():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 0px;
                visibility: hidden;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 0px;
                visibility: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
df=download()
# Tab de Planos
with tab[0]:
    #hide_sidebar()
    st.header("Escolha o Plano")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image('free.png',width=150)
        st.markdown("<div style='text-align: left;'>"
                    "<h3>Free</h3>"
                    "Base de 200mb máximo<br>"
                    "1 Modelo por usuário<br>"
                    "1 Inferência por dia<br>"
                    "Sem personalização<br>"
                    "Anúncios<br>"
                    "Sem previsão de demanda<br>"
                    "<b style='font-size: 24px;'>Grátis</b>"
                    "</div>", unsafe_allow_html=True)

    
    with col2:
        st.image('crown.png',width=150)
        st.markdown("<div style='text-align: left;'>"
                    "<h3>Premium</h3>"
                    "Sem limite de Base<br>"
                    "Múltiplos modelos<br>"
                    "Sem limite de inferência<br>"
                    "Completamente personalizável<br>"
                    "Sem anúncios<br>"
                    "Com previsão de demanda<br>"
                    "<b style='font-size: 24px;'>R$ 199,90/mês</b>"
                    "</div>", unsafe_allow_html=True)

    
    # Botão de teste grátis ocupando as duas colunas
    st.write('\n')
    st.button("Teste 1 semana grátis", use_container_width=True)

# Tab Cliente Normal
with tab[1]:
    st.sidebar.header("Interface")
    interface = st.sidebar.radio("Escolha a Interface:", ["Treino", "Inferência"], key="interface")

    # Interface de Treino
    if interface == "Treino":
        st.header("Interface de Treino - Cliente Normal")
        st.image("ads.png", caption="Anúncio", use_column_width='auto')
        
        st.file_uploader("Faça o upload da base de dados",key='1')
        
        show_toggle = st.checkbox("Mostrar como estruturar a base", key="normal_show_structure")
        if show_toggle:
            st.info("A base de dados deve estar em formato CSV, com a primeira linha contendo os nomes das colunas. Cada linha subsequente deve representar uma amostra. Não numere as amostras. A informação que você deseja prever deve ser a última coluna à direita.")
        
        task = st.radio("Defina a tarefa:", 
                        ["Classificação Binária", "Regressão", "Clustering"], key="normal_task")
        
        if task == "Classificação Binária":
            st.write("Classificação Binária: Classificação entre 2 grupos.")
        elif task == "Regressão":
            st.write("Regressão: Predição de valores numéricos.")
        elif task == "Clustering":
            st.write("Clustering: Agrupamento de dados similares.")
        
        if st.button("Treinar Modelo", key="normal_train_model"):
            st.image("ads.png", caption="Anúncio", use_column_width='auto')
            bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                bar.progress(i + 1)
            st.success('Modelo treinado com sucesso!')
            st.download_button("Baixar Modelo", data="modelo_treinado", file_name="modelo.pkl")

    # Interface de Inferência
    elif interface == "Inferência":
        st.header("Interface de Inferência - Cliente Normal")
        st.image("ads.png", caption="Anúncio", use_column_width='auto')
        
        st.file_uploader("Faça o upload da base de dados",key='2')
        
        show_toggle = st.checkbox("Mostrar como estruturar a base", key="normal_show_structure_inference")
        if show_toggle:
            st.info("A base de dados deve estar em formato CSV, com a primeira linha contendo os nomes das colunas. Cada linha subsequente deve representar uma amostra. Não numere as amostras. A informação que você deseja prever deve ser a última coluna à direita.")
        
        if st.button("Fazer Previsão", key="normal_predict"):
            st.image("ads.png", caption="Anúncio", use_column_width='auto')
            bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                bar.progress(i + 1)
            st.success('Previsão realizada com sucesso!')
            st.download_button("Baixar Resultados", data="resultados_previsao", file_name="resultados.csv")

# Tab Cliente Premium
with tab[2]:
    #st.sidebar.header("Interface")
    #interface = st.sidebar.radio("Escolha a Interface:", ["Treino", "Inferência"], key="interface")

    # Interface de Treino
    if interface == "Treino":
        st.header("Interface de Treino - Cliente Premium")
        
        st.file_uploader("Faça o upload da base de dados")
        
        show_data = st.toggle("Mostrar dados")
        if show_data:
            st.write(df[-756:-252])
            st.line_chart(df['Adj Close'][-756:-252])

        show_toggle = st.checkbox("Mostrar como estruturar a base", key="premium_show_structure")
        if show_toggle:
            st.info("A base de dados deve estar em formato CSV, com a primeira linha contendo os nomes das colunas. Cada linha subsequente deve representar uma amostra. Não numere as amostras. A informação que você deseja prever deve ser a última coluna à direita.")
        
        task = st.radio("Defina a tarefa:", 
                        ["Classificação Binária", "Regressão", "Clustering",'Previsão de demanda'], key="premium_task")
        
        if task == "Classificação Binária":
            st.write("Classificação Binária: Classificação entre 2 grupos.")
        elif task == "Regressão":
            st.write("Regressão: Predição de valores numéricos.")
        elif task == "Clustering":
            st.write("Clustering: Agrupamento de dados similares.")
        elif task=='Previsão de demanda':
            st.write('Previsão de demanda: prever valores numericos futuros.')

        customize_toggle = st.checkbox("Personalização", key="premium_customize")
        if customize_toggle:
            st.subheader("Personalização de Camadas")
            add_layer = st.button("Adicionar camada", key="premium_add_layer")
            if add_layer:
                with st.form(key="layer_form", clear_on_submit=True):
                    layer_type = st.selectbox("Tipo de Camada", ["Dense", "LSTM", "Conv"], key="layer_type")
                    if layer_type == "Dense":
                        activation = st.selectbox("Função de Ativação", ["relu", "sigmoid", "tanh"], key="dense_activation")
                        neurons = st.number_input("Número de Neurônios", min_value=1, step=1, key="dense_neurons")
                    elif layer_type == "LSTM":
                        activation = st.selectbox("Função de Ativação", ["relu", "sigmoid", "tanh"], key="lstm_activation")
                        units = st.number_input("Número de Unidades", min_value=1, step=1, key="lstm_units")
                    elif layer_type == "Conv":
                        filters = st.number_input("Número de Filtros", min_value=1, step=1, key="conv_filters")
                        kernel_size = st.number_input("Tamanho do Kernel", min_value=1, step=1, key="conv_kernel_size")
                        activation = st.selectbox("Função de Ativação", ["relu", "sigmoid", "tanh"], key="conv_activation")
                    submit_layer = st.form_submit_button("Adicionar")
                    if submit_layer:
                        st.success("Camada adicionada")
                        st.experimental_rerun()
            
            view_layers = st.checkbox("Ver Camadas", key="premium_view_layers")
            if view_layers:
                st.write("Lista de Camadas (exemplo)")
                # Exemplo de botões de camadas
                if st.button("Dense - 32 neurônios", key="view_dense_32"):
                    st.write("Configurações da camada Dense - 32 neurônios")
                    if st.button("Editar", key="edit_dense_32"):
                        with st.form(key="layer_form", clear_on_submit=True):
                            layer_type = st.selectbox("Tipo de Camada", ["Dense", "LSTM", "Conv"], key="layer_type")
                            if layer_type == "Dense":
                                activation = st.selectbox("Função de Ativação", ["relu", "sigmoid", "tanh"], key="dense_activation")
                                neurons = st.number_input("Número de Neurônios", min_value=1, step=1, key="dense_neurons")
                            elif layer_type == "LSTM":
                                activation = st.selectbox("Função de Ativação", ["relu", "sigmoid", "tanh"], key="lstm_activation")
                                units = st.number_input("Número de Unidades", min_value=1, step=1, key="lstm_units")
                            elif layer_type == "Conv":
                                filters = st.number_input("Número de Filtros", min_value=1, step=1, key="conv_filters")
                                kernel_size = st.number_input("Tamanho do Kernel", min_value=1, step=1, key="conv_kernel_size")
                                activation = st.selectbox("Função de Ativação", ["relu", "sigmoid", "tanh"], key="conv_activation")
                            submit_layer = st.form_submit_button("Adicionar")
                            if submit_layer:
                                st.success("Camada adicionada")
                                st.experimental_rerun()   
                    if st.button("Excluir", key="delete_dense_32"):
                        st.write("Camada excluída")
        
            st.subheader("Personalização do Método Compile")
            if st.button("Personalizar compile", key="premium_compile"):
                with st.form(key="compile_form", clear_on_submit=True):
                    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], key="compile_optimizer")
                    loss = st.selectbox("Loss", ["mse", "binary_crossentropy", "categorical_crossentropy"], key="compile_loss")
                    metrics = st.multiselect("Metrics", ["accuracy", "mae", "mse"], key="compile_metrics")
                    submit_compile = st.form_submit_button("Salvar")
                    if submit_compile:
                        st.success("Configurações de compile salvas")
                        st.experimental_rerun()
            
            st.subheader("Personalização do Método Fit")
            if st.button("Personalizar treino", key="premium_fit"):
                with st.form(key="fit_form", clear_on_submit=True):
                    epochs = st.number_input("Número de Épocas", min_value=1, step=1, key="fit_epochs")
                    batch_size = st.number_input("Tamanho do Batch", min_value=1, step=1, key="fit_batch_size")
                    validation_split = st.slider("Validation Split", 0.0, 1.0, step=0.1, key="fit_validation_split")
                    submit_fit = st.form_submit_button("Salvar")
                    if submit_fit:
                        st.success("Configurações de treino salvas")
                        st.experimental_rerun()

        if st.button("Treinar Modelo", key="premium_train_model"):
            bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                bar.progress(i + 1)
            st.success('Modelo treinado com sucesso!')
            st.download_button("Baixar Modelo", data="modelo_treinado", file_name="modelo.pkl")

    # Interface de Inferência
    elif interface == "Inferência":
        st.header("Interface de Inferência - Cliente Premium")
        
        st.file_uploader("Faça o upload da base de dados")
        
        show_data = st.toggle("Mostrar dados")
        if show_data:
            st.write(df[-252:])
            st.line_chart(df['Adj Close'][-252:])

        show_toggle = st.checkbox("Mostrar como estruturar a base", key="premium_show_structure_inference")
        if show_toggle:
            st.info("A base de dados deve estar em formato CSV, com a primeira linha contendo os nomes das colunas. Cada linha subsequente deve representar uma amostra. Não numere as amostras. A informação que você deseja prever deve ser a última coluna à direita.")
        
        model_choice = st.selectbox("Escolher Modelo", ["Modelo 1", "Modelo 2", "Modelo 3"], key="premium_model_choice")
        
        if st.button("Fazer Previsão", key="premium_predict"):
            bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                bar.progress(i + 1)
            st.success('Previsão realizada com sucesso!')
            st.download_button("Baixar Resultados", data="resultados_previsao", file_name="resultados.csv")
with tab[3]:
    st.header("Documentação")
    
    st.subheader("Introdução")
    st.write("""
        Bem-vindo ao Predict AI! Este site oferece serviços de treinamento e inferência de modelos de redes neurais
        para diversas tarefas. Você pode fornecer a base de dados para treinamento e configurar os hiperparâmetros 
        de treinamento conforme suas necessidades.
    """)

    st.subheader("Como Funciona")
    st.write("""
        1. **Escolha do Plano**: Selecione entre o plano Free ou Premium conforme suas necessidades.
        2. **Upload da Base de Dados**: Faça o upload da sua base de dados em formato CSV.
        3. **Configuração de Tarefas e Hiperparâmetros**: Escolha a tarefa de aprendizado e configure os hiperparâmetros 
        de treinamento, como número de camadas, funções de ativação, etc.
        4. **Treinamento do Modelo**: Inicie o treinamento do modelo e acompanhe a barra de progresso.
        5. **Inferência**: Utilize o modelo treinado para fazer previsões com novos dados.
    """)

    st.subheader("Camadas de Rede Neural")
    
    st.write("""
        **Dense (Totalmente Conectada)**:
        - Cada neurônio nesta camada está conectado a todos os neurônios da camada anterior.
        - Usada principalmente para aprendizado supervisionado.
        - Hiperparâmetros:
          - **Neurônios**: Número de unidades na camada.
          - **Função de Ativação**: Define a saída de cada neurônio (por exemplo, relu, sigmoid, tanh).
    """)

    st.write("""
        **LSTM (Long Short-Term Memory)**:
        - Camada especial para aprendizado de sequências e séries temporais.
        - Pode armazenar informações por longos períodos.
        - Hiperparâmetros:
          - **Unidades**: Número de unidades de memória.
          - **Função de Ativação**: Define a saída de cada unidade (por exemplo, relu, sigmoid, tanh).)
    """)

    st.write("""
        **Conv (Convolucional)**:
        - Camada utilizada para processamento de imagens.
        - Aplica filtros para extrair características das imagens.
        - Hiperparâmetros:
          - **Filtros**: Número de filtros de convolução.
          - **Tamanho do Kernel**: Tamanho da janela do filtro.
          - **Função de Ativação**: Define a saída de cada filtro (por exemplo, relu, sigmoid, tanh).
    """)

    st.subheader("Funções de Ativação")
    st.write("""
        **Função de Ativação**:
        - Define a saída de um neurônio em uma rede neural.
        - Introduz não-linearidade no modelo, permitindo aprender representações complexas dos dados.
    """)

    st.write("""
        **ReLU (Rectified Linear Unit)**:
        - Função linear que retorna o valor de entrada se for positivo, caso contrário retorna zero.
        - Comumente usada em redes neurais profundas devido à sua simplicidade e eficiência computacional.
    """)
    st.image('relu.png',use_column_width=True)
    st.write("""
        **Sigmoid**:
        - Função sigmoide que mapeia qualquer valor real para o intervalo (0, 1).
        - Utilizada principalmente em problemas de classificação binária.
        - ![Sigmoid](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)
    """)

    st.write("""
        **Tanh (Tangente Hiperbólica)**:
        - Função que mapeia qualquer valor real para o intervalo (-1, 1).
        - Utilizada para centralizar os dados, o que pode ajudar no aprendizado.

    """)
    st.image('tanh.png',use_column_width=True)
    st.subheader("Hiperparâmetros de Treinamento")
    
    st.write("""
        **Optimizer (Optimizador)**:
        - Algoritmo usado para ajustar os pesos do modelo e minimizar a função de perda.
        - Exemplos: adam, sgd, rmsprop.
    """)

    st.write("""
        **Loss (Função de Perda)**:
        - Medida de quão bem o modelo está se saindo durante o treinamento.
        - Exemplos: mse (erro quadrático médio), binary_crossentropy (entropia cruzada binária), categorical_crossentropy (entropia cruzada categórica).
    """)

    st.write("""
        **Metrics (Métricas)**:
        - Utilizadas para avaliar a performance do modelo.
        - Exemplos: accuracy (acurácia), mae (erro absoluto médio), mse (erro quadrático médio).
    """)

    st.write("""
        **Epochs (Épocas)**:
        - Número de vezes que o modelo verá todo o dataset durante o treinamento.
    """)

    st.write("""
        **Batch Size (Tamanho do Lote)**:
        - Número de amostras que o modelo verá antes de atualizar os pesos.
    """)

    st.write("""
        **Validation Split (Divisão de Validação)**:
        - Porcentagem do dataset usada para validação durante o treinamento.
    """)

    st.subheader("Fluxo de Trabalho")
    st.write("""
        1. **Upload da Base de Dados**:
           - A base de dados deve estar em formato CSV.
           - A primeira linha deve conter os nomes das colunas.
           - Cada linha subsequente deve representar uma amostra.
        
        2. **Configuração da Tarefa**:
           - Selecione a tarefa de aprendizado (Classificação Binária, Regressão, Clustering, Time Series Forecasting).
        
        3. **Personalização do Modelo**:
           - Adicione e configure camadas conforme necessário.
           - Personalize o método compile e fit.
        
        4. **Treinamento**:
           - Inicie o treinamento do modelo e acompanhe a barra de progresso.
        
        5. **Inferência**:
           - Utilize o modelo treinado para fazer previsões com novos dados.
    """)

    st.write("""
        Para mais informações, acesse a [documentação do TensorFlow](https://www.tensorflow.org/).
    """)