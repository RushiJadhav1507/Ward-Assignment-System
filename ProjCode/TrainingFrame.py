import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
import numpy as np
from ResultFrame import ResultFrame;
import globaldata
from tkinter.scrolledtext import ScrolledText
import re,os
from collections import defaultdict

folderpath=""

def main():
   
    program = TrainingFrame()
    program.w.mainloop()
    
class TrainingFrame:
    def __init__(self, root):
        self.w = root
        self.w = root
        Tk_Width = 850
        Tk_Height = 550
 
         #calculate coordination of screen and window form
        positionRight = int( self.w.winfo_screenwidth()/2 - Tk_Width/2 )
        positionDown = int( self.w.winfo_screenheight()/2 - Tk_Height/2 )

        # Set window in center screen with following way.
        self.w.geometry("{}x{}+{}+{}".format(850,600,positionRight, positionDown))
        self.f1 = tk.Frame(width=1150, height=200, background="gray")
        
        self.f1.pack(fill="both", expand=True, padx=40, pady=20)
       
        self.w.title("Training ...")
        self.w['bg']='#D8F2F1'
        self.f1['bg']='#D8F2F1'
        self.UI()

    def UI(self):
        style = ttk.Style()

        

        style.configure("TButton", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        b2 = ttk.Button(self.w, text="Training",style="TButton")
        b2.place(x=50, y=20,w=150,h=30)
        b2['command'] = lambda: self.process()

        style.configure("TButton", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        b3 = ttk.Button(self.w, text="Next",style="TButton")
        b3.place(x=240, y=20,w=150,h=30)
        b3['command'] = lambda: self.open_window(ResultFrame)        

        style.configure("BW.TLabel", foreground="black", background="#D8F2F1",font=('Helvetica', 24))
        ttk.Style().configure('pad.Label', padding='5 1 1 1')
        self.strt2=tk.StringVar()
        self.lbl1 = ttk.Label(self.w, textvariable=self.strt2 ,style="BW.TLabel")
        self.lbl1.place(x=50, y=90,w=650,h=120)
        
    def process(self):
        #self.strt2.set("Training process has been started")
        messagebox.showinfo("Result", "Training is underprocess ... Wait for few minutes!") 
        from DDPG import Test
        t=Test(data=globaldata.DF_admission_record)
        globaldata.rewards=t.process(data=globaldata.DF_admission_record) 
        messagebox.showinfo("Result", "Training has been done") 
        globaldata.rewards=0   
        self.strt2.set("Training process has been completed")

    def open_window(self,Win_class):
            self.w.withdraw()
            global win
            win=tk.Toplevel(self.w)
            Win_class(win)

    



if __name__ == "__main__":
    main()
