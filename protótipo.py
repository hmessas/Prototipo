import streamlit as st
import pandas as pd
import time

st.title('Predict AI (Protótipo)')

with st.sidebar:
    st.header('Opções')
    obj = st.radio('Objetivo',['Treinar','Prever'])

if obj == 'Treinar':
    st.write('Envie sua base de dados')
    st.file_uploader('Upload ')
    estr = st.toggle('Como minha base deve ser estruturada?')
    if estr:
        st.write('1. Lorem Ipsum')
        st.write('2. Lorem Ipsum')
        st.write('3. Lorem Ipsum')
    st.write('Descreva seu modelo')
    st.radio('Tarefa',['Regressão','Classificação Binária','Classificação Multi-Categórica','Lorem Ipsum'])
    detalhes = st.toggle('Personalização do modelo (Usuários Premium)')

    if detalhes:
        st.number_input('Número de camadas')
        st.number_input('Número de epochs')
        st.write('Lorem Ipsum')
    treinar = st.button('Treinar Modelo')
    if treinar:
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.03)
            bar.progress(i+1)
        a = st.write('Modelo treinado!')
        def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")
        df=convert_df(pd.DataFrame())
        st.download_button('Download as Keras file',df,file_name='Resultados.csv')
else:
    st.write('Envie sua base de dados')
    st.file_uploader('Upload ')
    estr = st.toggle('Como minha base deve ser estruturada?')
    if estr:
        st.write('1. Lorem Ipsum')
        st.write('2. Lorem Ipsum')
        st.write('3. Lorem Ipsum')
    treinar = st.button('Fazer previsão')
    if treinar:
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.03)
            bar.progress(i+1)
        st.write('Previsões feitas!')

        def convert_df(df):
        
            return df.to_csv().encode("utf-8")
        df=convert_df(pd.DataFrame())
        st.download_button('Download as CSV file',df,file_name='Resultados.csv')