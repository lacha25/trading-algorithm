
# ----- Fonction de décision d'achat -----
# C'est l'unique fonction de décision d'achat ou de vente, basée sur l'ensemble des retours des différents indicateurs
# La décision est prise en fonction de la valeur de 'signe', étant la variable gardant une trace de tous les retours des indicateurs
from Signaux import *
from analysisTools.YfAnalysis import f_yf_analysis

# Si le programme s'est actualisé plus de 5 fois (donc a tourné au moins 5 minutes), on calcule la valeur du signal d'achat en prenant en compte la valeur précédente
# On utilise pour cela la même formule que pour l'EMA
def f_vendre_ou_acheter(signal): 
  if not signal:
    return "wait", 0.0

  recent = signal[-5:]
  if len(recent) == 5:
    k_5 = 2 / (5 + 1)
    ema = recent[0]
    for value in recent[1:]:
      ema = value * k_5 + ema * (1 - k_5)
    signe = round(ema, 3)
  else:
    signe = float(recent[-1])

  if signe >= 0.5:
    return "buy", round(min(abs(signe), 1.0) * 100, 2)
  if signe <= -0.5:
    return "sell", round(min(abs(signe), 1.0) * 100, 2)
  return "wait", 100.0

#compute global signal: 
def f_update_signal(prix,signal,ent):
  if prix is None or len(prix) < 15:
    return signal

  coeff_ichi, coeff_rsi, coeff_yf = 1.0, 1.0, 1.0
  weighted_sum = 0.0
  max_signal = 0.0

  try:
    weighted_sum += coeff_ichi * float(f_signal_ichimoku(prix))
    max_signal += coeff_ichi
  except Exception:
    pass

  try:
    weighted_sum += coeff_rsi * float(f_signal_RSI(prix))
    max_signal += coeff_rsi
  except Exception:
    pass

  # External component is optional; keep strategy alive when the API is unavailable.
  try:
    yf_signal = f_yf_analysis(ent)
    if yf_signal is not None:
      weighted_sum += coeff_yf * float(yf_signal)
      max_signal += coeff_yf
  except Exception:
    pass

  if max_signal == 0:
    return signal

  newsignal = weighted_sum / max_signal
  signal.append(float(newsignal))
  if len(signal) > 5:
    signal.pop(0)
  return signal
    
def f_gain_potentiel(prixB,prixS,ent):
  buy_price = prixB[ent] if isinstance(prixB, dict) else prixB
  gain=round(((prixS/buy_price)-1)*100,2)
  return gain
  
