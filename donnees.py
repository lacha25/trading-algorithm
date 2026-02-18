""" ----- Import des données -----
Cette section du programme est composée de définitions de fonctions permettant
l'import des données financières et leur gestion.
"""

import contextlib
import io
from typing import List

import pandas as pd


# Convertisseur des données en fichier .csv avec un format approprié
def f_data_to_CSV(d):
  fichier = open(d, "r")
  tab = fichier.readlines()
  tab.pop(0)
  content = []
  for i in tab:
    content.append(i.split())
  fichier.close()
  return content


# ----- Fonction d'import des données -----
# Cette fonction utilise defeatbeta-api pour récupérer les données de prix.
# the shape of data_prix is data_prix[i][0] = close price, data_prix[i][1] = open price
def import_donnees(entreprise):
  try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      from defeatbeta_api.data.ticker import Ticker
      data = Ticker(entreprise).price()
  except Exception:
    return None

  if data is None or data.empty:
    return None

  required_cols = {"open", "close", "report_date"}
  if not required_cols.issubset(set(data.columns)):
    return None

  df = data[["report_date", "open", "close"]].copy()
  df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
  df["open"] = pd.to_numeric(df["open"], errors="coerce")
  df["close"] = pd.to_numeric(df["close"], errors="coerce")
  df = df.dropna(subset=["report_date", "open", "close"]).sort_values("report_date")
  if df.empty:
    return None

  # Keep recent bars to limit runtime while preserving enough history
  df = df.tail(1000)

  close_series = df["close"].round(6)
  open_series = df["open"].round(6)

  close_series.to_csv("DataC.csv", index=False, header=[entreprise])
  open_series.to_csv("DataO.csv", index=False, header=[entreprise])

  data_prix: List[List[float]] = []
  for close_value, open_value in zip(close_series, open_series):
    data_prix.append([float(close_value), float(open_value)])
  return data_prix
