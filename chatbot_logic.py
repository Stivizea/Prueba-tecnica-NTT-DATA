import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

def inicializar_agente_cafe(df):
    # Configuramos Gemini. 'temperature=0' es vital para análisis de datos precisos.
    # 
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=st.secrets["API KEY DE GOOGLE"] 
    )
    
    # Creamos el agente
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=False, 
        allow_dangerous_code=True # Requerido para ejecutar el código Python que genera Gemini
    )
    return agent