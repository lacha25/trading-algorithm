""" ----- Import des données -----
Cette section du programme est composée de définitions de fonctions permettant
l'import des données financières et leur gestion.
"""

import contextlib
import io
import time
from typing import List, Optional

import pandas as pd
import yfinance as yf


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


def _normalize_price_df(data: pd.DataFrame) -> Optional[pd.DataFrame]:
  if data is None or data.empty:
    return None

  cols_by_lower = {str(col).lower(): col for col in data.columns}
  open_col = cols_by_lower.get("open")
  close_col = cols_by_lower.get("close")
  if open_col is None or close_col is None:
    return None

  df = data[[open_col, close_col]].copy()
  df.columns = ["open", "close"]
  df["open"] = pd.to_numeric(df["open"], errors="coerce")
  df["close"] = pd.to_numeric(df["close"], errors="coerce")
  df = df.dropna(subset=["open", "close"])
  if df.empty:
    return None
  return df


def _load_from_defeatbeta(entreprise: str) -> Optional[pd.DataFrame]:
  try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      from defeatbeta_api.data.ticker import Ticker
      data = Ticker(entreprise).price()
  except Exception:
    return None

  if not isinstance(data, pd.DataFrame) or data.empty:
    return None

  if "report_date" in data.columns:
    data = data.copy()
    data["report_date"] = pd.to_datetime(data["report_date"], errors="coerce")
    data = data.dropna(subset=["report_date"]).sort_values("report_date")
  return _normalize_price_df(data)


def _load_from_yfinance(entreprise: str) -> Optional[pd.DataFrame]:
  for attempt in range(3):
    try:
      with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        data = yf.download(
            tickers=entreprise,
            period="5d",
            interval="1m",
            group_by="column",
            auto_adjust=True,
            prepost=True,
            threads=True,
            progress=False,
        )
    except Exception:
      data = pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
      if entreprise in data.columns.get_level_values(-1):
        data = data.xs(entreprise, axis=1, level=-1)
      else:
        data = data.groupby(level=0, axis=1).first()

    normalized = _normalize_price_df(data)
    if normalized is not None and not normalized.empty:
      return normalized
    time.sleep(0.5 + attempt)
  return None


# ----- Fonction d'import des données -----
# Cette fonction tente defeatbeta-api puis yfinance pour récupérer les données de prix.
# the shape of data_prix is data_prix[i][0] = close price, data_prix[i][1] = open price
def import_donnees(entreprise):
  df = _load_from_defeatbeta(entreprise)
  if df is None or df.empty:
    df = _load_from_yfinance(entreprise)

  if df is None or df.empty:
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
