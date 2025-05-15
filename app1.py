import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import joblib

# Charger le modèle
model = joblib.load("ranfor.pkl")

# Charger les données pour la visualisation

st.title("🌦️ Prédiction de la pluie")

# Saisie utilisateur
temp = st.number_input("Température (°C)", step=0.1)
hum = st.number_input("Humidité (%)", step=0.1)
wind = st.number_input("Vitesse du vent", step=0.1)
cloud = st.number_input("Couverture nuageuse (%)", step=0.1)
press = st.number_input("Pression (hPa)", step=0.1)

if st.button("🔮 Prédire"):
    data = np.array([[temp, hum, wind, cloud, press]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🌧️ Il va pleuvoir.")
    else:
        st.success("☀️ Pas de pluie.")

if st.button("Analyse et Visualisation"):

    st.subheader("Chargement des données")
    try:
        df = pd.read_csv("Weather_forecast_data.csv")
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())

        st.subheader("Statistiques descriptives")
        st.write(df.describe())
        st.write("Nombre de valeurs manquantes :")
        st.write(df.isnull().sum())

        st.subheader("Carte de chaleur des corrélations")
        fig1, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig1)
  
        st.subheader("Distribution des colonnes")
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
         st.write(f"📊 Histogramme de **{col}**")
         fig3, ax = plt.subplots()
         sns.histplot(df[col], bins=30, kde=True, ax=ax, color='skyblue', edgecolor='black')
         ax.set_title(f"Distribution de {col}")
         st.pyplot(fig3)

             # Visualisation du résultat sous forme de diagramme
            # Sélection des colonnes utilisées pour la prédiction
        input_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
        
        # Vérifier que les colonnes existent
        if all(col in df.columns for col in input_cols):
            X = df[input_cols]
            
            # Prédictions de probabilité
            proba = model.predict_proba(X)[0]  # première ligne uniquement

            st.subheader("📉 Visualisation des probabilités")

            fig4, ax = plt.subplots()
            ax.bar(['Pas de pluie', 'Pluie'], proba, color=['salmon', 'blue'],edgecolor='black')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probabilité")
            ax.set_title("Probabilités de prédiction")
            st.pyplot(fig4)
        else:
            st.warning("Les colonnes nécessaires pour la prédiction ne sont pas présentes dans le fichier.")
      

    except Exception as e:
     st.error(f"Erreur lors du chargement ou de l'analyse des données : {e}")