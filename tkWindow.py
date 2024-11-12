#OBSOLETE


import tkinter as tk #pour fenetre graphique
from decision import * #importe les differents fichiers
from donnees import *
from MACD import *
from tkWindow import *
import datetime #pour la date
import pytz


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


date = datetime.datetime.now(tz=pytz.timezone('US/Eastern'))
heure = date.hour
jour = date.weekday()
fen = tk.Tk()
fen.title('∑asyMoney®')
fen.geometry("600x400")
canvas = tk.Canvas(fen, width=500, height=300,background='white')
canvas.place(x=100,y=100)
canvas.pack()


def tkinter():
  
  # Calcul  de la date et l'heure américaine grâce aux modules datetime et pytz. 
  # Renvoie un message d'erreur si la bourse est fermée (comprend à la fois la bourse américaine)

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
    main_rep_tk()
    
    b1 = tk.Button(fen, text="Retirer une entreprise",bd='5', command=retirer_ent)
    b1.place(x=50,y=310)
    b2 = tk.Button(fen, text="Quitter",bd='5', command=fen.destroy)
    b2.place(x=255 ,y=310)
    b3 = tk.Button(fen, text="Ajouter une entreprise",bd='5', command=ajout_ent)
    b3.place(x=360,y=310)
    '''b4 = tk.Button(fen, text="Voir",bd='5', command=f_afficher_courbe('AAPL')) #TEST
    b4.place(x=380,y=150)'''
  fen.mainloop()

def main_rep_tk():
    
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

