import pickle
import streamlit as st
import pandas as pd


# importation des datasets de test :
atest = pd.read_csv('atest.csv')                   # original
atest_encoded = pd.read_csv('atest_encoded.csv')   # encodé

# importation des datasets d'entraînement :
at = pd.read_csv('at.csv')                   # original


def get_client():
    idClient = st.selectbox('Choisir un client', atest.index)
    return idClient

# Sélection du client
idClient = get_client()

# Calcul des indicateurs de comparaison :
bacsd = int(at['B_AMT_CREDIT_SUM_DEBT'].mean())
bacmo = int(at['B_AMT_CREDIT_MAX_OVERDUE'].mean())
baa = at['B_AMT_ANNUITY'].mean()
cfm = at['CNT_FAM_MEMBERS'].mean()
bbms = at['BB_MAX_STATUS'].mean()
bbms = round(bbms,2)

# Affichage données client :
st.subheader('Informations client')
cols = st.beta_columns(4)
cols[0].write(f'Age: {atest.iloc[idClient]["DAYS_BIRTH"]}')
cols[1].write(f'Statut Fam.: {atest.iloc[idClient]["NAME_FAMILY_STATUS"]}')
cols[2].write(f'Nbre famille: {atest.iloc[idClient]["CNT_FAM_MEMBERS"]}')
cols[3].write(f'Type Education: {atest.iloc[idClient]["NAME_EDUCATION_TYPE"]}')

cols = st.beta_columns(4)
cols[0].write(f'Voiture: {atest.iloc[idClient]["FLAG_OWN_CAR"]}')
cols[1].write(f'Type Revenu: {atest.iloc[idClient]["NAME_INCOME_TYPE"]}')
cols[2].write(f'Revenu: {atest.iloc[idClient]["AMT_INCOME_TOTAL"]}')

st.subheader('Crédit')
cols = st.beta_columns(4)
cols[0].write(f'Type Emprunt: {atest.iloc[idClient]["NAME_CONTRACT_TYPE"]}')
cols[1].write(f'Montant Crédit: {atest.iloc[idClient]["AMT_CREDIT"]}')
cols[2].write(f'Annuité: {atest.iloc[idClient]["AMT_ANNUITY"]}')

st.subheader('Passif')
cols = st.beta_columns(4)
cols[0].write(f'Dettes ext.: {int(atest.iloc[idClient]["B_AMT_CREDIT_SUM_DEBT"])}')
cols[1].write(f'Dettes ext. Moy: {bacsd}')
cols[2].write(f'Max souf: {atest.iloc[idClient]["B_AMT_CREDIT_MAX_OVERDUE"]}')
cols[3].write(f'Max souf Moy: {bacmo}')
cols = st.beta_columns(4)
cols[0].write(f'Max DPD.: {int(atest.iloc[idClient]["BB_MAX_STATUS"])}')
cols[1].write(f'MAX DPD Moy: {bbms}')

my_expander = st.beta_expander(label='Facteurs influents')
with my_expander:
    st.image("SHAP.PNG", width=None)
    clicked = st.button('Click me!')

# importer le model
model = pickle.load(open('rf_for_deployment','rb'))

# Prévision : appliquer le modèle sur le client choisi
prev = model.predict(pd.DataFrame(atest_encoded, index=[idClient]))
proba = model.predict_proba(pd.DataFrame(atest_encoded, index=[idClient]))[0][prev[0]].round(4)

st.subheader('Prévision de défaut de crédit')   # seuil opt=0.55
if proba > 0.55:
    st.write("Défaut")
else:
    st.write("Sans Défaut")

cols = st.beta_columns(2)
cols[0].write(f'Probabilité: {proba}')
cols[1].write(f'Client considéré en défaut si proba > 0.55')

