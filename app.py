import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import torch
import torch.nn as nn
import joblib

# Configuración básica de la app
st.set_page_config(
    page_title="High Garden Coffee | Analítica Global CRM",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos de las gráficas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper")

# ----------------------------------------------------
# Definición de la arquitectura de PyTorch
# ----------------------------------------------------
class GlobalLSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(GlobalLSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ----------------------------------------------------
# Funciones de carga de datos y modelos (con caché para que vuele)
# ----------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_parquet('coffee_db.parquet')
    year_cols = [col for col in df.columns if '/' in col]
    
    # Pasamos el dataset a formato long para que sea procesable
    df_long = df.melt(
        id_vars=['Country', 'Coffee type'],
        value_vars=year_cols,
        var_name='Year',
        value_name='Consumption'
    )
    
    # Limpiamos la columna del año (ej. '1990/91' -> 1990)
    df_long['Year'] = df_long['Year'].str.split('/').str[0].astype(int)
    df_long = df_long.sort_values(by=['Country', 'Coffee type', 'Year']).reset_index(drop=True)
    return df, df_long

@st.cache_resource
def load_ml_assets():
    try:
        scaler = joblib.load('consumption_scaler.gz')
        model = GlobalLSTMForecaster(hidden_size=64, num_layers=2)
        model.load_state_dict(torch.load('global_lstm_best.pth', map_location=torch.device('cpu')))
        model.eval() 
        return model, scaler
    except FileNotFoundError:
        return None, None

@st.cache_data(show_spinner=False)
def generate_all_forecasts(_model, _scaler, df):
    """Genera las predicciones base para todos los países de una sola pasada."""
    if _model is None or _scaler is None:
        return None
        
    future_records = []
    
    # Inferencia sobre cada serie de tiempo individual
    for (country, ctype), group in df.groupby(['Country', 'Coffee type']):
        ts_data = group.sort_values('Year')
        
        # Necesitamos al menos 5 años de datos históricos para generar la secuencia
        if len(ts_data) >= 5:
            historical_vals = ts_data['Consumption'].values
            current_seq_scaled = _scaler.transform(historical_vals[-5:].reshape(-1, 1))
            current_tensor = torch.tensor(current_seq_scaled, dtype=torch.float32).unsqueeze(0)
            
            future_preds_scaled = []
            with torch.no_grad():
                for _ in range(5):
                    pred = _model(current_tensor)
                    future_preds_scaled.append(pred.item())
                    pred_tensor = pred.unsqueeze(1)
                    current_tensor = torch.cat((current_tensor[:, 1:, :], pred_tensor), dim=1)
            
            # Revertimos el escalado para tener los valores reales en tazas
            future_preds = _scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
            
            for i, year in enumerate([2020, 2021, 2022, 2023, 2024]):
                future_records.append({
                    'Country': country,
                    'Coffee type': ctype,
                    'Year': year,
                    'Consumption': future_preds[i]
                })
                
    return pd.DataFrame(future_records)

# Cargamos todo a memoria apenas inicia la app
raw_df, df_long = load_and_prepare_data()
lstm_model, global_scaler = load_ml_assets()

# Generamos las predicciones globales por debajo de cuerda
with st.spinner("Inicializando el motor de Deep Learning..."):
    df_predicted = generate_all_forecasts(lstm_model, global_scaler, df_long)

# ----------------------------------------------------
# Interfaz: Navegación del Sidebar
# ----------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924514.png", width=100) 
    st.title("CRM High Garden")
    st.markdown("---")
    mode = st.radio(
        "Navegación",
        [
            "Resumen Ejecutivo", 
            "Concentración de Mercado (Pareto)", 
            "Mapa de Calor Global", 
            "Matriz de Saturación de Mercado", 
            "📈 Predicciones con Deep Learning (LSTM)",
            "🤖 Asistente de IA Generativa"
        ]
    )
    st.markdown("---")
    st.caption("Desarrollado para la prueba técnica de NTT DATA")

# ----------------------------------------------------
# Vista 1: Resumen Ejecutivo
# ----------------------------------------------------
if mode == "Resumen Ejecutivo":
    st.header("Resumen Ejecutivo: Tendencias Globales de Consumo")
    st.markdown("Una vista a nivel macro del consumo doméstico de café desde 1990 hasta 2019.")
    
    col1, col2, col3 = st.columns(3)
    latest_year = df_long['Year'].max()
    total_consump_latest = df_long[df_long['Year'] == latest_year]['Consumption'].sum()
    total_consump_prev = df_long[df_long['Year'] == latest_year - 1]['Consumption'].sum()
    growth = ((total_consump_latest - total_consump_prev) / total_consump_prev) * 100
    
    col1.metric("Consumo Global Total (2019)", f"{total_consump_latest:,.0f} Tazas", f"{growth:.2f}% Anual")
    col2.metric("Mercados Activos Evaluados", f"{df_long['Country'].nunique()} Países")
    col3.metric("Periodo de Datos", "30 Años")
    
    st.markdown("### Cambio en el Consumo por Tipo de Café a Nivel Global")
    
    type_trend = df_long.groupby(['Year', 'Coffee type'])['Consumption'].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    type_trend.plot(ax=ax, linewidth=2.5, cmap='tab10') 
    ax.set_ylabel('Consumo Total (Tazas)')
    ax.set_xlabel('Año')
    ax.legend(title='Tipo de Café')
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Análisis más granular al final del resumen
    st.header("Análisis Específico de Mercado (Deep-Dive)")
    st.markdown("Selecciona un país específico para aislar su historial de consumo desglosado por tipo de café.")
    
    selected_country = st.selectbox("Selecciona un Mercado", df_long['Country'].unique())
    country_data = df_long[df_long['Country'] == selected_country]
    
    st.subheader(f"Tendencias Históricas: {selected_country}")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=country_data, x='Year', y='Consumption', hue='Coffee type', marker="o", ax=ax2)
    ax2.set_ylabel('Consumo (Tazas)')
    ax2.set_xlabel('Año')
    st.pyplot(fig2)
    
    st.dataframe(country_data.pivot_table(index='Year', columns='Coffee type', values='Consumption').tail(5), width='stretch')

# ----------------------------------------------------
# Vista 2: Concentración de Mercado (Pareto)
# ----------------------------------------------------
elif mode == "Concentración de Mercado (Pareto)":
    st.header("Concentración de Mercado: Análisis de Pareto")
    st.markdown("Identificando los mercados clave que impulsan la mayor parte del consumo global.")
    
    df_2019 = df_long[df_long['Year'] == 2019].groupby('Country')['Consumption'].sum().sort_values(ascending=False)
    pareto_df = pd.DataFrame({'Consumption': df_2019})
    pareto_df['Cumulative_Pct'] = pareto_df['Consumption'].cumsum() / pareto_df['Consumption'].sum() * 100
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    top_n = 20
    ax1.bar(pareto_df.index[:top_n], pareto_df['Consumption'][:top_n], color='#2c3e50')
    ax1.set_ylabel('Consumo (Tazas)', color='#2c3e50', weight='bold')
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(pareto_df.index[:top_n], rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.plot(pareto_df.index[:top_n], pareto_df['Cumulative_Pct'][:top_n], color='#e74c3c', marker='o', ms=6, linewidth=2)
    ax2.set_ylabel('Porcentaje Acumulado %', color='#e74c3c', weight='bold')
    ax2.axhline(80, color='gray', linestyle='--', alpha=0.7, label='Umbral del 80%')
    st.pyplot(fig)
    
    st.info("💡 **Insight Clave:** Una pequeña fracción de los países representa más del 80% del consumo doméstico total. High Garden Coffee debería concentrar sus esfuerzos de optimización logística en estas top 10 regiones.")

# ----------------------------------------------------
# Vista 3: Mapa Geoespacial
# ----------------------------------------------------
elif mode == "Mapa de Calor Global":
    st.header("Intensidad del Consumo Global (2019)")
    with st.spinner('Renderizando los datos geoespaciales...'):
        df_2019 = df_long[df_long['Year'] == 2019].groupby('Country')['Consumption'].sum().reset_index()
        world = gpd.read_file('ne_110m_admin_0_countries.zip')
        
        merge_col = 'name' 
        for col in ['NAME', 'name', 'SOVEREIGNT', 'sovereignt', 'ADMIN', 'admin']:
            if col in world.columns:
                merge_col = col
                break

        # Arreglando un par de nombres para que hagan match perfecto con el shapefile
        name_map = {
            'Bolivia (Plurinational State of)': 'Bolivia',
            'Democratic Republic of Congo': 'Dem. Rep. Congo',
            'Viet Nam': 'Vietnam',
            'Lao People\'s Democratic Republic': 'Laos'
        }
        df_2019['Country'] = df_2019['Country'].replace(name_map)
        world_merged = world.merge(df_2019, left_on=merge_col, right_on='Country', how='left')
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        world_merged.plot(
            column='Consumption', ax=ax, legend=True,
            legend_kwds={'label': "Consumo Doméstico (2019)", 'orientation': "horizontal", 'fraction': 0.046, 'pad': 0.04},
            cmap='OrRd', missing_kwds={'color': '#eeeeee'}
        )
        ax.set_axis_off()
        st.pyplot(fig)

# ----------------------------------------------------
# Vista 4: Matriz de Saturación
# ----------------------------------------------------
elif mode == "Matriz de Saturación de Mercado":
    st.header("Matriz Histórica de Saturación de Mercado (2014-2019)")
    st.markdown("""
    Esta matriz clasifica los mercados comparando su **Volumen Total (2019)** contra su **Tasa de Crecimiento a 5 años (2014-2019)**. 
    * **Mercados Saturados (Abajo a la derecha):** Alto volumen, bajo crecimiento (o negativo). El mercado ya tocó techo.
    * **Estrellas Emergentes (Arriba a la derecha):** Alto volumen, alto crecimiento. Hay que meterle toda la ficha acá.
    * **Crecimiento de Nicho (Arriba a la izquierda):** Bajo volumen, alto crecimiento. Posibles líderes en el futuro.
    """)
    
    df_2014 = df_long[df_long['Year'] == 2014].groupby('Country')['Consumption'].sum()
    df_2019 = df_long[df_long['Year'] == 2019].groupby('Country')['Consumption'].sum()
    
    sat_df = pd.DataFrame({'Vol_2014': df_2014, 'Vol_2019': df_2019}).dropna()
    sat_df = sat_df[(sat_df['Vol_2014'] > 0) & (sat_df['Vol_2019'] > 0)]
    sat_df['CAGR_5YR'] = (sat_df['Vol_2019'] / sat_df['Vol_2014']) ** (1/5) - 1
    sat_df['CAGR_5YR_Pct'] = sat_df['CAGR_5YR'] * 100
    
    vol_threshold = sat_df['Vol_2019'].median()
    cagr_threshold = sat_df['CAGR_5YR_Pct'].median()
    
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.scatterplot(data=sat_df, x='Vol_2019', y='CAGR_5YR_Pct', s=120, color='#3498db', edgecolor='black', alpha=0.6, ax=ax)
    
    ax.axvline(vol_threshold, color='red', linestyle='--', alpha=0.5)
    ax.axhline(cagr_threshold, color='red', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    
    ax.set_title('Matriz de Saturación de Mercado Histórica', weight='bold', fontsize=16)
    ax.set_xlabel('Volumen de Consumo Total en 2019 (Escala Logarítmica)', weight='bold')
    ax.set_ylabel('Tasa de Crecimiento Anual Compuesto a 5 Años (%)', weight='bold')
    
    try:
        from adjustText import adjust_text
        top_vol = sat_df.nlargest(15, 'Vol_2019').index.tolist()
        top_growth = sat_df.nlargest(5, 'CAGR_5YR_Pct').index.tolist()
        bottom_growth = sat_df.nsmallest(5, 'CAGR_5YR_Pct').index.tolist()
        countries_to_label = list(set(top_vol + top_growth + bottom_growth))
        
        texts = [ax.text(sat_df.loc[c, 'Vol_2019'], sat_df.loc[c, 'CAGR_5YR_Pct'], c, fontsize=9, weight='bold') for c in countries_to_label]
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), ax=ax)
    except ImportError:
        pass
    
    st.pyplot(fig)
    
    st.subheader("Mercados Saturados (Zonas de Alerta)")
    st.markdown("Estos países consumen una cantidad masiva de café pero están mostrando estancamiento o un declive en sus compras. Las estrategias de marketing aquí deberían enfocarse en la **retención de usuarios y premiumización** en lugar de gastar plata en tratar de adquirir nuevo volumen.")
    
    saturated_markets = sat_df[(sat_df['Vol_2019'] > vol_threshold) & (sat_df['CAGR_5YR_Pct'] < cagr_threshold)].sort_values(by='Vol_2019', ascending=False)
    
    display_df = saturated_markets[['Vol_2019', 'CAGR_5YR_Pct']].copy()
    display_df['CAGR_5YR_Pct'] = display_df['CAGR_5YR_Pct'].apply(lambda x: f"{x:.2f}%")
    display_df.rename(columns={'Vol_2019': 'Volumen Total (2019)', 'CAGR_5YR_Pct': 'Tasa de Crecimiento a 5 Años'}, inplace=True)
    
    st.dataframe(display_df, width='stretch')

# ----------------------------------------------------
# Vista 5: Deep Learning 
# ----------------------------------------------------
elif mode == "📈 Predicciones con Deep Learning (LSTM)":
    st.header("Analítica Predictiva con Deep Learning")
    st.markdown("Acá usamos un modelo **Auto-Regresivo Global LSTM en PyTorch**, entrenado de manera simultánea en todos los mercados para capturar la dinámica temporal macroeconómica y evitar el sobreajuste clásico que vemos en los modelos estadísticos tradicionales.")
    
    if lstm_model is None or df_predicted is None:
        st.error("Pilas, no encontramos los archivos del modelo. Asegúrate de tener 'global_lstm_best.pth' y 'consumption_scaler.gz' en la misma carpeta del proyecto.")
    else:
        # --- SECCIÓN 1: PRONÓSTICO GLOBAL ---
        st.subheader("1. Pronóstico del Volumen Global (2020-2024)")
        
        global_hist = df_long.groupby('Year')['Consumption'].sum()
        global_pred = df_predicted.groupby('Year')['Consumption'].sum()
        
        fig_global, ax_global = plt.subplots(figsize=(12, 4))
        ax_global.plot(global_hist.index[-10:], global_hist.values[-10:], marker='o', color='#2c3e50', linewidth=2.5, label='Total Histórico')
        
        # Tiramos la línea punteada conectando el pasado y el futuro
        ax_global.plot([global_hist.index[-1], global_pred.index[0]], [global_hist.values[-1], global_pred.values[0]], color='#8e44ad', linestyle='--', linewidth=2.5)
        ax_global.plot(global_pred.index, global_pred.values, marker='X', markersize=8, color='#8e44ad', linestyle='--', linewidth=2.5, label='Pronóstico Total LSTM')
        
        ax_global.set_title("Salida Agregada de la Red Predictiva Global", weight='bold')
        ax_global.set_ylabel("Total de Tazas Consumidas")
        ax_global.set_xlabel("Año")
        ax_global.set_xticks(np.concatenate([global_hist.index[-10:], global_pred.index]))
        ax_global.axvspan(global_hist.index[-1], global_pred.index[-1], color='#8e44ad', alpha=0.05)
        ax_global.legend()
        st.pyplot(fig_global)
        
        st.markdown("---")
        
        # --- SECCIÓN 2: MATRIZ DE SATURACIÓN PREDICTIVA ---
        st.subheader("2. Matriz de Saturación de Mercado Predictiva (Pronóstico a 2024)")
        st.markdown("Esta matriz nos da una visión adelantada: identifica qué mercados van a acelerarse o estancarse en los próximos 5 años, dándole a High Garden Coffee la ventaja de mover sus cadenas de suministro proactivamente.")
        
        df_2019_hist = df_long[df_long['Year'] == 2019].groupby('Country')['Consumption'].sum()
        df_2024_pred = df_predicted[df_predicted['Year'] == 2024].groupby('Country')['Consumption'].sum()
        
        pred_sat_df = pd.DataFrame({'Vol_2019': df_2019_hist, 'Vol_2024': df_2024_pred}).dropna()
        pred_sat_df = pred_sat_df[(pred_sat_df['Vol_2019'] > 0) & (pred_sat_df['Vol_2024'] > 0)]
        pred_sat_df['CAGR_5YR_PRED'] = (pred_sat_df['Vol_2024'] / pred_sat_df['Vol_2019']) ** (1/5) - 1
        pred_sat_df['CAGR_5YR_PRED_Pct'] = pred_sat_df['CAGR_5YR_PRED'] * 100
        
        vol_thresh_pred = pred_sat_df['Vol_2024'].median()
        cagr_thresh_pred = pred_sat_df['CAGR_5YR_PRED_Pct'].median()
        
        fig_pred_sat, ax_pred_sat = plt.subplots(figsize=(14, 9))
        sns.scatterplot(data=pred_sat_df, x='Vol_2024', y='CAGR_5YR_PRED_Pct', s=120, color='#8e44ad', edgecolor='black', alpha=0.6, ax=ax_pred_sat)
        
        ax_pred_sat.axvline(vol_thresh_pred, color='red', linestyle='--', alpha=0.5)
        ax_pred_sat.axhline(cagr_thresh_pred, color='red', linestyle='--', alpha=0.5)
        ax_pred_sat.set_xscale('log')
        
        ax_pred_sat.set_title('Matriz de Saturación Predictiva (2019 -> 2024)', weight='bold', fontsize=16)
        ax_pred_sat.set_xlabel('Volumen Total Pronosticado para 2024 (Escala Log)', weight='bold')
        ax_pred_sat.set_ylabel('Tasa de Crecimiento Pronosticada a 5 Años (%)', weight='bold')
        
        try:
            from adjustText import adjust_text
            top_vol_p = pred_sat_df.nlargest(15, 'Vol_2024').index.tolist()
            top_growth_p = pred_sat_df.nlargest(5, 'CAGR_5YR_PRED_Pct').index.tolist()
            bottom_growth_p = pred_sat_df.nsmallest(5, 'CAGR_5YR_PRED_Pct').index.tolist()
            labels_p = list(set(top_vol_p + top_growth_p + bottom_growth_p))
            
            texts_p = [ax_pred_sat.text(pred_sat_df.loc[c, 'Vol_2024'], pred_sat_df.loc[c, 'CAGR_5YR_PRED_Pct'], c, fontsize=9, weight='bold') for c in labels_p]
            adjust_text(texts_p, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), ax=ax_pred_sat)
        except ImportError:
            pass
        
        st.pyplot(fig_pred_sat)
        
        st.markdown("---")

        # --- SECCIÓN 3: PAÍSES ESPECÍFICOS ---
        st.subheader("3. Pronóstico Aislado por País")
        col1, col2 = st.columns(2)
        with col1:
            pred_country = st.selectbox("Seleccionar Mercado a Pronosticar", df_long['Country'].unique(), index=0)
        with col2:
            available_types = df_long[df_long['Country'] == pred_country]['Coffee type'].unique()
            pred_type = st.selectbox("Seleccionar Tipo de Café", available_types)
            
        ts_data = df_long[(df_long['Country'] == pred_country) & (df_long['Coffee type'] == pred_type)].sort_values('Year')
        ts_pred = df_predicted[(df_predicted['Country'] == pred_country) & (df_predicted['Coffee type'] == pred_type)].sort_values('Year')
        
        if len(ts_data) < 5 or ts_pred.empty:
            st.error("Faltan datos históricos para poder armar la ventana de secuencia.")
        else:
            historical_years = ts_data['Year'].values
            historical_vals = ts_data['Consumption'].values
            future_years = ts_pred['Year'].values
            future_preds = ts_pred['Consumption'].values
            
            fig_ind, ax_ind = plt.subplots(figsize=(12, 5))
            ax_ind.plot(historical_years[-10:], historical_vals[-10:], marker='o', color='#2c3e50', linewidth=2.5, label='Datos Históricos')
            
            ax_ind.plot([historical_years[-1], future_years[0]], [historical_vals[-1], future_preds[0]], color='#e74c3c', linestyle='--', linewidth=2.5)
            ax_ind.plot(future_years, future_preds, marker='X', markersize=8, color='#e74c3c', linestyle='--', linewidth=2.5, label='Pronóstico LSTM')
            
            ax_ind.set_title(f"Pronóstico de la Red a 5 Años: {pred_type} en {pred_country}", weight='bold')
            ax_ind.set_ylabel("Tazas Consumidas")
            ax_ind.set_xlabel("Año")
            ax_ind.set_xticks(np.concatenate([historical_years[-10:], future_years]))
            ax_ind.axvspan(historical_years[-1], future_years[-1], color='#e74c3c', alpha=0.05)
            ax_ind.legend()
            st.pyplot(fig_ind)

# ----------------------------------------------------
# Vista 6: GenAI 
# ----------------------------------------------------
elif mode == "🤖 Asistente de IA Generativa":
    st.header("High Garden AI: Agente para Análisis de Datos")
    st.markdown("Esta pestaña cumple con el requisito del **BONUS** de la prueba, demostrando cómo los LLMs pueden transformar datos totalmente estáticos en un motor de consultas interactivo.")
    
    st.info("Propuesta de Arquitectura: LangChain + Pandas Dataframe Agent + API de OpenAI/Gemini. El agente tendría contexto del dataframe limpio y podría escribir scripts en Python por detrás para responderle a los stakeholders al instante.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola!, Soy la IA de High Garden. Pregúntame lo que necesites sobre nuestros datos de consumo doméstico."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Por ejemplo: ¿Qué país tuvo la mayor tasa de crecimiento entre 2010 y 2019?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            st.markdown("*(Simulación de respuesta de IA)* Revisando los datos, **Vietnam** mostró la mayor Tasa de Crecimiento Anual Compuesto (CAGR) para el consumo de Robusta en ese periodo. Esto va muy de la mano con la brutal expansión que han tenido en sus capacidades de tostado doméstico.")
        st.session_state.messages.append({"role": "assistant", "content": "*(Simulación de respuesta de IA)* Revisando los datos, **Vietnam** mostró la mayor Tasa de Crecimiento Anual Compuesto (CAGR) para el consumo de Robusta en ese periodo. Esto va muy de la mano con la brutal expansión que han tenido en sus capacidades de tostado doméstico."})
        
        #CODIGO PARA LA IMPLEMENTACION DEL AGENTE DE IA (NO HECHO DEBIDO A QUE GOOGLE CLOUD ME ESTABA DANDO PROBLEMAS CON LAS TOKENS DE GEMINI)

##elif mode == "🤖 Asistente de IA Generativa":
##    st.header("High Garden AI: Consultas con Gemini 1.5")
##    st.markdown("Este asistente utiliza **Google Gemini** para analizar el dataset en tiempo real. Puedes pedirle cálculos, comparaciones o resúmenes de los datos.")

    # Verificamos que la API Key esté configurada
##    if "GOOGLE_API_KEY" not in st.secrets:
##        st.error("Falta la configuración de 'GOOGLE_API_KEY' en los secretos de Streamlit.")
##    else:
##        from langchain_google_genai import ChatGoogleGenerativeAI
#      from langchain_experimental.agents import create_pandas_dataframe_agent

        # Inicialización del agente en el estado de la sesión
#       if "agente_gemini" not in st.session_state:
#            with st.spinner("Conectando con Gemini..."):
#                llm = ChatGoogleGenerativeAI(
#                    model="gemini-1.5-flash",
#                    temperature=0,
#                    google_api_key=st.secrets["GOOGLE_API_KEY"]
#                )
#                st.session_state.agente_gemini = create_pandas_dataframe_agent(
#                    llm, 
#                    df_long, 
#                    verbose=False, 
#                    allow_dangerous_code=True
 #               )

 #       # Historial de conversación
 #       if "messages" not in st.session_state:
 #           st.session_state.messages = [{"role": "assistant", "content": "¡Listo! Tengo el dataset cargado. ¿Qué quieres que analicemos hoy?"}]

 #       for message in st.session_state.messages:
 #           with st.chat_message(message["role"]):
 #               st.markdown(message["content"])

 #       if prompt := st.chat_input("Ej: ¿Cuál es el CAGR promedio de los países en el cuadrante de Niche Growth?"):
 #           st.session_state.messages.append({"role": "user", "content": prompt})
 #           with st.chat_message("user"):
 #               st.markdown(prompt)
            
 #           with st.chat_message("assistant"):
 #               with st.spinner("Gemini está analizando los datos..."):
 #                   try:
                        # Ejecución de la consulta
 #                       respuesta = st.session_state.agente_gemini.run(prompt)
#                        st.markdown(respuesta)
 #                       st.session_state.messages.append({"role": "assistant", "content": respuesta})
 #                   except Exception as e:
 #                       st.error(f"Hubo un error al procesar la consulta: {str(e)}") ##