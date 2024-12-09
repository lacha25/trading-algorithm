from analysisTools.RSI import f_RSI
from analysisTools.MACD import *
from analysisTools.Ichimoku import*

# Les 2 fonctions d'interprétation des indicateurs financiers à suivre sont basées sur le même principe: une combinaison de cas possibles, menant dans chaque cas à une modification de la variable signal.
# Cette variable est dans chaque cas renvoyée par la fonction

# Fonction d'interpretation de l'indicateur Ichimoku
def f_signal_ichimoku(prix): 
  kijun=f_Kijun(prix)
  tenkan=f_Tenkan(prix)
  SSA=f_SSA(kijun,tenkan)
  SSB=f_SSB(prix)
  chk=f_chikou(prix)

  #Le prix arrête de stagner et s'apprete a monter ou a descendre
  if chk[0] > kijun[25] and chk[0] > tenkan[25] and chk[0] > SSA[25] and chk[0] > SSB[25]:
    verif = 1
  elif chk[0] < kijun[25] and chk[0] < tenkan[25] and chk[0] < SSA[25] and chk[0] < SSB[25]:
    verif = -1  
  else:
    verif=0
  signal=0
  #le prix croise la courbe "kijun"
  if kijun[0]>=kijun[1] and kijun[0]>=kijun[2] and  prix[-2][0]>kijun[1]>prix[-2][1] and verif==1 :
    signal+=0.75
  elif prix[-2][0]<kijun[1]<prix[-2][1] and kijun[0]<kijun[1] and verif==-1:
    signal-=0.75
  #le prix traverse le nuage
  elif (prix[-1][0]>SSA[0] and  prix[-2][0]>SSA[1]>prix[-2][1]) or (prix[-1][0]>SSB[0] and  prix[-2][0]>SSB[1]>prix[-2][1]):
    signal+=0.25
  elif(prix[-1][0]<SSA[0] and prix[-2][0]<SSA[1]<prix[-2][1] ) or (prix[-1][0]<SSB[0] and prix[-2][0]<SSB[1]<prix[-2][1]):
    signal-=0.25
  #le prix est au dessus du nuage/en dessous
  elif prix[-1][0]>SSA[0] and prix[-1][0]>SSB[0]:
    signal+=0.1
  elif prix[-1][0]<SSA[0] and prix[-1][0]<SSB[0]:
    signal-=0.1
  return signal

def f_signal_RSI(prix):
  RSI=f_RSI(prix)

  return RSI[0]*2 -1 
# Fonction d'interpretation de l'indicateur MACD avec l'interpretation du RSI
# Les interpretations sont communes car ces 2 indicateurs sont complémentaires
def f_signal_MACD(prix): 
  MACD_1=f_MACD_1(prix)
  MACD_2=f_MACD_2(prix)
  MACD_h=f_MACD_histogram(MACD_1,MACD_2)
  up_trend,down_trend=0,0
  signal=0
  survendu,surachete=0,0

  ##########
  ##TODO
  ############
  return signal
