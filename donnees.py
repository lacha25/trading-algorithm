""" ----- Import des données -----
Cette section du programme est composée de définitions de fonctions permettant l'import des données financières souhaitées et leur gestion (conversion en .csv)"""

# Import absolument fondamental du module yfinance, qui est la source unique de l'ensemble des données financières utilisées
# Ce module est utilisé dans la fonction : import_donnees(entreprise)
import yfinance as yf 

# Convertisseur des données à importer en fichier .csv avec un format approprié
def f_data_to_CSV(d):
  fichier = open(d, "r") 
  tab=fichier.readlines()
  tab.pop(0)
  content=[]
  for i in tab:
    content.append(i.split())
  fichier.close
  return content


# ----- Fonction d'import des données -----
# Cette fonction utilise le module importé yfinance pour récupérer les données de prix des titres d'intérêt
# De nombreux paramètres sont modifiables dans celle-ci comme les entreprises ou titres d'intérêt, la période d'inport et la fréquence
# the shape of data_prix is data_prix[i][0] = close price, data_prix[i][1] = open price
#  data_prix[i+1][0] = close price of the next day, data_prix[0]=close price of the first day
def import_donnees(entreprise):
  data = yf.download(  # or pdr.get_data_yahoo(...
          # tickers list or string as well
          tickers = entreprise,

          # use "period" instead of start/end
          # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
          # (optional, default is '1mo')
          period = "1d",

          # fetch data by interval (including intraday if period < 60 days)
          # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
          # (optional, default is '1d')
          interval = "1m",

          # group by ticker (to access via data['SPY'])
          # (optional, default is 'column')
          group_by = 'column',

          # adjust all OHLC automatically
          # (optional, default is False)
          auto_adjust = True,

          # download pre/post regular market hours data
          # (optional, default is False)
          prepost = True,

          # use threads for mass downloading? (True/False/Integer)
          # (optional, default is True)
          threads = True,

          # proxy URL scheme use use when downloading?
          # (optional, default is None)
          proxy = None
      )
  data["Close"].to_csv(r'DataC.csv', index = False)
  data_prix=f_data_to_CSV('DataC.csv')
  data["Open"].to_csv(r'DataO.csv', index = False)
  data_prixO=f_data_to_CSV('DataO.csv')
  for i in range(len(data_prix)) :
    data_prix[i][0]=round(float(data_prix[i][0]),6)
    data_prix[i].append(round(float(data_prixO[i][0]),6))
  
 
  return data_prix