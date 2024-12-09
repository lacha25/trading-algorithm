
from decision import * #importe les differents fichiers
from donnees import *
import datetime #pour la date
import pytz
import streamlit as st
import time 
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt

# Algorithme venant extraire les données de Yahoo Finance pour les exploiter. Il fonctionne de la façon suivante: lors de l’exécution on rentre un certain nombre de noms d’entreprises et pour chacune de celles-ci le programme renvoie, en d’actualisant à une fréquence donnée, si le titre est à acheter, à vendre ou s’il faut attendre. Cette indication est accompagnée d'un indice de fiabilité en pourcentage permettant plus précision sur la situation financière actuelle.
# Ce projet fonctionne en TEMPS REEL, et les décisions sont actualisées à chaque minute (dans la configuration actuelle)

def check_market_hours():
    """Check if the market is open."""
    date = datetime.datetime.now(tz=pytz.timezone('US/Eastern'))
    heure = date.hour
    jour = date.weekday()
    return 9 <= heure <= 16 and jour < 5


#Programme principal qui se répète et crée la fenetre graphique

# création de la fentre et du canvas 
# Initialize session state variables if they don't exist
if 'listent' not in st.session_state:
    st.session_state['listent'] = ["AAPL", "SNAP", "AI"]
if 'signal' not in st.session_state:
    st.session_state['signal'] = {company: [] for company in st.session_state['listent']}
if 'data_list' not in st.session_state:
    st.session_state['data_list'] = []
#NOT MANDATORY FOR THE PROGRAM
#pB = prix d'achat pour traquer les gains potentiels
if 'pB' not in st.session_state:
    st.session_state['pB'] = {ent: 0 for ent in st.session_state['listent']}
if 'gain_en_prcnt' not in st.session_state:
    st.session_state['gain_en_prcnt'] = {ent: 0 for ent in st.session_state['listent']}


st.title("∑asyMoney® Trading Dashboard")
st_autorefresh(interval=60 * 1000, key="datarefresh")
# Display current time
st.write("Current Time (Eastern): ", datetime.datetime.now(tz=pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S"))

# Check market hours
market_open = check_market_hours()
if not market_open:
    st.warning("The market is currently closed. Please try again during market hours (9 AM - 4 PM ET).")
st.session_state['data_list'].clear()  # Clear `data_list` each time to prevent duplication

for ent in st.session_state['listent']:
    data = import_donnees(ent)  
    if data is not None:
        # Extract closing prices
        data_prix = [row[0] for row in data]

        # Update signals (here is done without extraction of closing price)
        st.session_state['signal'][ent] = f_update_signal(data, st.session_state['signal'][ent],ent)
        b_ou_s = f_vendre_ou_acheter(st.session_state['signal'][ent])

        # FOR DISPLAY PURPOSE
        val = b_ou_s[0] + " (" + str(b_ou_s[1]) + "%)"
        if b_ou_s[0] == 'buy':
            st.session_state['pB'][ent] = data[-1][0]
        if b_ou_s[0] == 'sell' and st.session_state['pB'][ent] != 0:
            pS = data[-1][0]
            st.session_state['gain_en_prcnt'][ent] += f_gain_potentiel(st.session_state['pB'][ent], pS, ent)
            st.session_state['pB'][ent] = 0

        # Add data entry for the table
        data_entry = {
            'Company': ent,
            'Current Price ($)': data_prix[-1],
            'Decision': val,
            'Gain Potential (%)': st.session_state['gain_en_prcnt'][ent]
        }
        st.session_state['data_list'].append(data_entry)
    else:
        st.warning(f"Could not retrieve data for {ent}")

# Display the Summary Table Above the Charts
if st.session_state['data_list']:
    df = pd.DataFrame(st.session_state['data_list'])
    st.subheader("Summary Table")
    st.dataframe(df)

# Display Each Company's Price Chart Below the Table
for ent in st.session_state['listent']:
    data = import_donnees(ent)
    if data is not None:
        data_prix = [row[0] for row in data]

        # Plot the price chart
        st.subheader(f"Price Chart for {ent}")
        min_value, max_value = min(data_prix), max(data_prix)

        plt.figure(figsize=(10, 5))
        plt.plot(data_prix, label="Price")
        plt.ylim(min_value, max_value)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)
        st.markdown("---")

# Sidebar Controls to Add or Remove Companies
st.sidebar.subheader("Manage Companies")
new_company = st.sidebar.text_input("Add a Company (Ticker Symbol):")
if st.sidebar.button("Add") and new_company:
    if new_company.upper() not in st.session_state['listent']:
        st.session_state['listent'].append(new_company.upper())
        st.session_state['pB'][new_company.upper()] = 0
        st.session_state['gain_en_prcnt'][new_company.upper()] = 0
        st.session_state['signal'][new_company.upper()] = []
    else:
        st.sidebar.warning("Company already exists.")

company_to_remove = st.sidebar.selectbox("Select Company to Remove", st.session_state['listent'])
if st.sidebar.button("Remove") and company_to_remove:
    st.session_state['listent'].remove(company_to_remove)
    st.session_state['pB'].pop(company_to_remove, None)
    st.session_state['gain_en_prcnt'].pop(company_to_remove, None)
    st.session_state['signal'].pop(company_to_remove, None)

# Results summary at the bottom of the page
"""
#not correct but easy to implement, to be done later 
st.header("Summary of Trading Decisions")
if st.session_state['gain_en_prcnt'] > 0:
    st.success(f"Overall Gain Potential: {st.session_state['gain_en_prcnt']}%")
elif st.session_state['gain_en_prcnt'] < 0:
    st.error(f"Overall Loss Potential: {st.session_state['gain_en_prcnt']}%")
else:
    st.info("No gain or loss at the moment.")
"""
# --- VERIFIER QUE CA MARCHE ---

#bugs/ameliorations:
#pas possible de mettre 0 entreprise 
#il y'a un nombre max d'entreprises qu'on peut mettre 
#il faut rentrer le bon nom d'entreprises sinon le programme crash 
#On peu facilement réduire le "poids" du programme 
#au lieu d'importer toutes les données on pourrait importer seulement celle nécessaire
#le signal Ichimoku n'est pas complet
#il faut vérifier nos indicateurs sur fenetre grphique, pas d'autre possibilité pour s'assurer du bon fonctionnement