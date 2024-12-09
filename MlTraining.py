from donnees import import_donnees
from analysisTools.CurveFitting import *

ent = 'AI'
dataAI = import_donnees(ent)
dataAAPL = import_donnees('AAPL')  
#f_MLP_CurveFitting_t(dataAI)
f_MLP_CurveFitting_e(dataAAPL)