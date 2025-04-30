# app.py
import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit.components.v1 import html

# ——— Page config & custom CSS ———
st.set_page_config(page_title="Model Predictor", layout="centered")
st.markdown("""
  <style>
    /* Background and font */
    .reportview-container { background-color: #F7FAFC; color: #2D3748; font-family: 'Segoe UI',sans-serif; }
    /* Button style */
    .stButton>button { background-color: #3182CE; color: white; border-radius: 8px; padding: .5em 1.5em; font-size:1rem; }
    /* Inputs and selects */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div>div { border:1px solid #CBD5E0; border-radius:6px; padding:.4em; }
    /* Container padding: top, right, bottom, left */
    .block-container { padding:1rem 3rem 1rem 3rem; }
    /* Tighten title margins */
    h1, .stTitle { margin-top: 0.5rem; margin-bottom: 0.5rem; }
  </style>
""", unsafe_allow_html=True)

# ——— Load model & clusters ———
@st.cache_resource
def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_clusters(path="clusters.csv"):
    df = pd.read_csv(path)
    return df.set_index("cluster_id")[['lat','lon']]

model = load_model()
clusters_df = load_clusters()

days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ——— UI Inputs ———
st.title("⚙️ Predict Your Target for All Clusters on Map")
st.write("Enter base features to predict demand across all clusters:")
col1, col2 = st.columns(2)
with col1:
    ft_1 = st.number_input("Demand at time T-1", key="ft_1", value=250, step=1)
    ft_2 = st.number_input("Demand at time T-2", key="ft_2", value=250, step=1)
    ft_3 = st.number_input("Demand at time T-3", key="ft_3", value=200, step=1)
    ft_4 = st.number_input("Demand at time T-4", key="ft_4", value=200, step=1)
with col2:
    ft_5    = st.number_input("Demand at time T-5", key="ft_5", value=200, step=1)
    weekday = st.selectbox("Weekday", options=days, key="weekday")
    exp_avg = st.number_input("Exp Avg", key="exp_avg", value=220, step=1)

# ——— Prediction callback for all clusters ———
def predict_all():
    wd_idx = days.index(st.session_state.weekday)
    records = []
    for cid, row in clusters_df.iterrows():
        records.append({
            'ft_1': st.session_state.ft_1,
            'ft_2': st.session_state.ft_2,
            'ft_3': st.session_state.ft_3,
            'ft_4': st.session_state.ft_4,
            'ft_5': st.session_state.ft_5,
            'lat': row['lat'],
            'lon': row['lon'],
            'weekday': wd_idx,
            'exp_avg': st.session_state.exp_avg,
            'cluster_id': cid
        })
    df_all = pd.DataFrame(records)
    df_all['demand'] = model.predict(df_all[[
        'ft_1','ft_2','ft_3','ft_4','ft_5','lat','lon','weekday','exp_avg'
    ]])
    st.session_state.df_all = df_all

# ——— Predict button ———
st.button("Predict for All Clusters", on_click=predict_all)

# ——— Display map of all clusters with demand labels ———
if 'df_all' in st.session_state:
    df_all = st.session_state.df_all
    m = folium.Map(location=[df_all['lat'].mean(), df_all['lon'].mean()], zoom_start=11)
    for _, r in df_all.iterrows():
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=8,
            color='blue', fill=True, fill_opacity=0.2
        ).add_to(m)
        folium.map.Marker(
            [r['lat'], r['lon']],
            icon=folium.DivIcon(html=f"<div style='font-size:14px; color:black'><b>{r['demand']:.0f}</b></div>")
        ).add_to(m)
    # Embed map with reduced height
    html(m._repr_html_(), height=450)

    # Table below map
    st.markdown("### Predicted Demand for All Clusters")
    st.dataframe(df_all[['cluster_id','demand']].set_index('cluster_id'))