
# ----- Fonction de décision d'achat -----
# C'est l'unique fonction de décision d'achat ou de vente, basée sur l'ensemble des retours des différents indicateurs
# La décision est prise en fonction de la valeur de 'signe', étant la variable gardant une trace de tous les retours des indicateurs
from Signaux import *
import math #pour la valeur absolue
from RSI import *
from MACD import *


# Si le programme s'est actualisé plus de 5 fois (donc a tourné au moins 5 minutes), on calcule la valeur du signal d'achat en prenant en compte la valeur précédente
# On utilise pour cela la même formule que pour l'EMA
def f_vendre_ou_acheter(signal): 
  if len(signal)==5:
    list_signe=[]
    list_signe.append(signal[0])
    k_5=2/(5+1)

    for i in range(4):
      list_signe.append(round(signal[i+1]*k_5+list_signe[i]*(1-k_5),3))

    signe=list_signe[-1]

    if signe>=0.5:
      return "buy",(round(abs(signe/4*100),2))
    elif -0.5<signe<0.5 :
      return "wait",100
    else:
      return "sell",(round(abs(signe/4*100),2))
  else:
    if signal[-1]>=0.5:
      return "buy",(round(abs(signal[-1]/4*100),2))
    elif -0.5<signal[-1]<0.5 :
      return "wait",100
    else:
      return "sell",(round(abs(signal[-1]/4*100),2))


def f_update_signal(prix,signal):
  signal.append(f_signal_achat_ichimoku(prix)+f_decision_achat_MACD_et_RSI(prix))
  if len(signal)>5:
    signal.remove(signal[1])
  return signal
    
def f_gain_potentiel(prixB,prixS,ent):
  gain=round(((prixB[ent]/prixS)-1)*100,2)
  return gain
  