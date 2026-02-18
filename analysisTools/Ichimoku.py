# ----- INDICATEUR ICHIMOKU -----
# L'indicateur Ichimoku est très complet : composé de 5 courbes celui-ci permet une analyse assez complète du prix et de son évolution
# Les 5 coubes le composant sont: Tenkan, Kijun, SSA, SSB et chikou et sont calculées repectivement par les 5 fonctions à suivre
# On utilise des SMA, dont le calcul n'est pas spécialement complexe (contrairement aux EMA, voir les détails dans le fichier: MACD.py)
# On a ensuite une dernière fonction dans cette partie qui permet d'interpreter les différents signaux renvoyés par l'indicateur

def f_Tenkan(prix):
  if len(prix) < 34:
    raise ValueError("Not enough price points for Tenkan.")
  list_tenkan=[]
  for j in range(26): 
    m_9=[]
    for i in range (1,10):
      m_9.append(float(prix[-(i+j)][0]))
    tenkan=(min(m_9)+max(m_9))/2
    list_tenkan.append(tenkan)
  return  list_tenkan


def f_Kijun(prix):
  if len(prix) < 52:
    raise ValueError("Not enough price points for Kijun.")
  list_kijun=[]
  for j in range(26):
    m_26=[]
    for i in range (1,27):
      m_26.append(float(prix[-(i+j)][0]))
    kijun=(min(m_26)+max(m_26))/2
    list_kijun.append(kijun)
  return list_kijun


def f_SSA(list_tenkan,list_kijun):
  list_SSA=[]
  for j in range(26):
    list_SSA.append((list_tenkan[j]+list_kijun[j])/2)
  return list_SSA


def f_SSB(prix):
  if len(prix) < 103:
    raise ValueError("Not enough price points for SSB.")
  list_SSB=[]
  for j in range(26):
    m_52=[]
    for i in range (1,53):
      m_52.append(float(prix[(-26-i-j)][0])) #Decale le prix de 26 chandelles dans le futur
    SSB=(min(m_52)+max(m_52))/2 
    list_SSB.append(SSB)
  return list_SSB


# Attention calculable uniquement sur periode n>26 chandelles
def f_chikou(prix): 
  if len(prix) < 30:
    raise ValueError("Not enough price points for Chikou.")
  list_chikou=[]
  for i in range(5):
    list_chikou.append(prix[-26-i][0]) #Decale le prix de 26 chandelles dans le passé
  return list_chikou
