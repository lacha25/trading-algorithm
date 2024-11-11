import tkinter as tk #pour fenetre graphique
from decision import * #importe les differents fichiers
from donnees import *
from MACD import *
import datetime #pour la date
import pytz

# Algorithme venant extraire les données de Yahoo Finance pour les exploiter. Il fonctionne de la façon suivante: lors de l’exécution on rentre un certain nombre de noms d’entreprises et pour chacune de celles-ci le programme renvoie, en d’actualisant à une fréquence donnée, si le titre est à acheter, à vendre ou s’il faut attendre. Cette indication est accompagnée d'un indice de fiabilité en pourcentage permettant plus précision sur la situation financière actuelle.
# Ce projet fonctionne en TEMPS REEL, et les décisions sont actualisées à chaque minute (dans la configuration actuelle)


# Fonction "principale" qui est répetée toutes les minutes: elle permet de renvoyer la décision ainsi que le gain_potentiel sur la fenetre tkinter. On ne peut pas la déplacer dans un autre fichier car on utilise global dans celle ci

def f_changement(x,y,ent,prix):
  global gain_en_prcnt  
  global pB
  global signal
  val=0
  val_var = tk.StringVar()
  val_var.set(str(val))
  val_gain = tk.StringVar()
  val_gain.set(str(gain_en_prcnt))
  # Affichage de la décision et du résultat sur le canvas on appelle la fonction sur b_ou_s au lieu de val uniquement pour améliorer l'affichage
  text_val = canvas.create_text(x, y, text=val_var.get(),fill='black')
  text_gain = canvas.create_text(440,270, text=(val_gain.get()),fill='black')
  
  signe=signal
  signal=f_liste_signe(prix,signe)
  b_ou_s=f_vendre_ou_acheter(signal)
  val=b_ou_s[0]+" ("+str(b_ou_s[1])+"%)"
  if b_ou_s[0]=='buy':
    pB[ent]=prix[-1][0]
  if b_ou_s[0]=='sell' and pB[ent]!= 0:
    pS=prix[-1][0]
    gain_en_prcnt=gain_en_prcnt+f_gain_potentiel(pB,pS,ent)
    pB[ent]=0

  val_var.set(str(val))
  val_gain.set(str(gain_en_prcnt))
  # Mise à jour du texte affiché :
  canvas.itemconfigure(text_val, text=val_var.get())
  canvas.itemconfigure(text_gain, text=val_gain.get())
  return


# Appel de la fonction d'affichage de la courbe de prix de l'entreprise
def f_afficher_courbe(ent):
  f_afficher_MACD(import_donnees(ent))
  return


 # Fonctions qui permettent d'ajouter ou d'enlever des entreprises a la liste qui elle est globale, il n'y a donc pas besoin de l'avoir en argument (on ajoute sur la console pour l'instant) 
  # Les boutons qui suivent appelle les fonctions d'au-dessus

def ajout_ent():
    global listent, pB
    entre = input("Rentrez l'entreprise a ajouter \n") 
    listent.append(entre)
    pB[entre] = 0
    print("Veuillez attendre quelques instants avant l'actualisation")
    return

def retirer_ent(): 
    global listent, pB
    entre = input("Rentrez l'entreprise a retirer\n")
    print("Veuillez attendre quelques instants avant l'actualisation")
    listent.remove(entre)
    del pB[entre]
    return


#Programme principal qui se répète et crée la fenetre graphique
def main_rep():
    
    global listent, gain_en_prcnt, pB
    
    canvas.delete('all')
    print("ca marche")
    canvas.create_text(133, 30, text='Entreprise',fill='black')
    canvas.create_text(233, 30, text='Prix',fill='black')
    canvas.create_text(346, 30, text='Décision (certitude)',fill='black')
    canvas.create_line(100, 45,400, 45)
    if gain_en_prcnt>0:
      canvas.create_rectangle(390,240,490,280,width=3,fill='lightgreen',outline='darkgreen')
    elif gain_en_prcnt<0:
      canvas.create_rectangle(390,240,490,280,width=3,fill='lightcoral',outline='red4')
    else:
      canvas.create_rectangle(390,240,490,280,width=3,fill='lightgrey',outline='darkgrey')
    canvas.create_text(440,250, text=('Résultats en %'))

    Y=60
    for ent in listent:
      data_prix=import_donnees(ent)
      f_changement(333,Y,ent,data_prix)
      canvas.create_line(280, Y-10,280, Y+20,width=2,fill='black')
      canvas.create_line(180,Y-10,180,Y+20,width=2,fill='black')
      canvas.create_text(233, Y, text=str(round(data_prix[-1][0],2))+"$",fill='black')
      canvas.create_text(133, Y, text=ent,fill='black')
    
      Y+=30
    fen.after(60000,main_rep)
    return


# ----- Programme pricipal -----

# création de la fentre et du canvas 
listent=["AAPL","SNAP","AI"]
fen = tk.Tk()
fen.title('∑asyMoney®')
fen.geometry("600x400")
pB = {ent: 0 for ent in listent}
gain_en_prcnt = 0
val =0
canvas = tk.Canvas(fen, width=500, height=300,background='white')
canvas.place(x=100,y=100)
canvas.pack()


# Calcul  de la date et l'heure américaine grâce aux modules datetime et pytz. 
# Renvoie un message d'erreur si la bourse est fermée (comprend à la fois la bourse américaine)
date = datetime.datetime.now(tz=pytz.timezone('US/Eastern'))
heure = date.hour
jour = date.weekday()

#---------------------FOR TESTING PURPOSES---------------------####


if False and (heure < 9 or heure > 16 or jour==5 or jour==6):
  btn = tk.Button(fen, text = 'Réessayer plus tard', bd = '5',command = fen.destroy)
  btn.pack(side = 'bottom')
  canvas.create_text(250,100, text='La bourse est actuellement fermée,',fill='black')
  canvas.create_text(250,120, text='Veuillez relancer le programme lorsque celle-ci sera ouverte.',fill='black')

# Si la bourse est ouverte, on crée une fonction qui se repete toutes les minutes.
# Celle ci s'actualise toutes en supprimant les anciennes données et graphismes pour ne pas qu'ils se superposent.On appelle dans celle ci la fonction f_changement qui affiche la décision
# la liste d'entreprise est intégrée dans cette fonction car elle permet de rajouter ou enlever une entreprise quand le bouton est pressé
else:
  signal=[]
  main_rep()
  
  b1 = tk.Button(fen, text="Retirer une entreprise",bd='5', command=retirer_ent)
  b1.place(x=50,y=310)
  b2 = tk.Button(fen, text="Quitter",bd='5', command=fen.destroy)
  b2.place(x=255 ,y=310)
  b3 = tk.Button(fen, text="Ajouter une entreprise",bd='5', command=ajout_ent)
  b3.place(x=360,y=310)
  '''b4 = tk.Button(fen, text="Voir",bd='5', command=f_afficher_courbe('AAPL')) #TEST
  b4.place(x=380,y=150)'''
fen.mainloop()





# --- VERIFIER QUE CA MARCHE ---

#bugs/ameliorations:
#pas possible de mettre 0 entreprise 
#il y'a un nombre max d'entreprises qu'on peut mettre 
#il faut rentrer le bon nom d'entreprises sinon le programme crash 
#On peu facilement réduire le "poids" du programme 
#au lieu d'importer toutes les données on pourrait importer seulement celle nécessaire
#le signal Ichimoku n'est pas complet
#il faut vérifier nos indicateurs sur fenetre grphique, pas d'autre possibilité pour s'assurer du bon fonctionnement