 # ----- INDICATEUR MACD -----
# MACD : Moving Average Convergence Divergence
# L'indicateur est composé de 2 courbes: la courbe MACD (MACD_1) et la courbe de signal (MACD_2)
# Ces courbes sont des EMA (Exponential Moving Average) donc des moyennes mobiles 'pondérées' ce qui complique énormément la lisibilité
# Ces EMA sont opposées aux SMA (Simple Moving Average) qui n'est qu'une moyenne arithmétique des n derniers prix
# Sous certaines conditions (interprétées par la fonction: f_decision_achat_MACD), on peut déduire des signaux d'achat ou de vente"""


#Import de matplotlib permettant un affichage potentiel des courbes de prix mais aussi d'indicateurs financiers avec leurs courbes respectives
import matplotlib.pyplot as plt
import os
os.environ['MPLCONFIGDIR'] = "/C:/"

# Formule à la base du MACD_1 : EMA [today] = (Price [today] x K) + (EMA [yesterday] x (1 – K)) avec K = 2 ÷ (N + 1) ici N=26 et 12
# NB : le marqueur temporel "today" permet un meilleure compréhension, mais ici la période considérée est de 1 minute
# On peut simplement en retenir que l'EMA se trouve en utilisant l'EMA précédent ajouté au prix actuel
def f_MACD_1(prix):
  EMA_12,EMA_26,list_MACD_1=[],[],[]
  k_12,k_26=2/(12+1),2/(26+1) # coefficient d'adoucicement

  EMA_12.append(prix[-12][0]) # -n n=12
  c=0
  for i in range (11,0,-1): # range(n-1,0,-1) avec n le nombre de jours
    EMA_12.append(round((prix[-i][0])*k_12+(EMA_12[c])*(1-k_12),3))
    c+=1
    
  EMA_26.append(prix[-26][0]) # -n+1 n=26
  c=0
  for i in range (25,0,-1): # range(n-1,0,-1) avec n le nombre de jours
    EMA_26.append(round((prix[-i][0])*k_26+(EMA_26[c])*(1-k_26),3))
    c+=1
  for i in range (12): #min n
    list_MACD_1.append(EMA_12[i]-EMA_26[i])
  return list_MACD_1

# MACD Signal line (EMA) : coube de signal aussi basée sur une moyenne mobile EMA
def f_MACD_2(prix):  
  list_MACD_2=[]
  k_9=2/(9+1)
  list_MACD_2.append(prix[-9][0]) # -n n=9
  c=0
  for i in range (8,0,-1): # range(n-1,0,-1) avec n le nombre de jours
    list_MACD_2.append(round((prix[-i][0])*k_9+(list_MACD_2[c])*(1-k_9),3))
    c+=1
  return list_MACD_2

# MACD Histogram : représentation particulière du MACD sous forme d'un histogramme sur une fenetre indépendante du prix
def f_MACD_histogram(MACD_1,MACD_2):
  list_MACD_h=[]
  for i in range(10):
    list_MACD_h.append(MACD_1[-i]-MACD_2[-i])
  return list_MACD_h

# Fonction d'affichage par matplotlib des 2 courbes composant le MACD : MACD_1 et MACD_2
# Cette fonction rajoute beaucoup de difficulté quand au décalage des courbes entre elles
def f_afficher_MACD(prix):
  MACD_1=f_MACD_1(prix)
  MACD_2=f_MACD_2(prix)

  plt.xlabel('Temps')
  plt.ylabel('Prix $')
  plt.title('Courbe de prix')
  y=[x for (x,y) in prix[-50:]]
  x =range(len(y))
  #x1=range(len(MACD_1))
  #x2=range(len(MACD_2))
  plt.plot(x, y,'r.-')
  plt.plot(x1, MACD_1,'b.-')
  plt.plot(x2, MACD_2,'y.-')
  plt.grid()
  plt.show()
  return