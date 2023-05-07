import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
import numpy as np;
import globaldata
from tkinter.scrolledtext import ScrolledText
import re,os
from collections import defaultdict

folderpath=""

def main():
   
    program = ResultFrame()
    program.w.mainloop()
    
class ResultFrame:
    def __init__(self, root):
        self.w = root
        self.w = root
        Tk_Width = 1250
        Tk_Height = 550
 
         #calculate coordination of screen and window form
        positionRight = int( self.w.winfo_screenwidth()/2 - Tk_Width/2 )
        positionDown = int( self.w.winfo_screenheight()/2 - Tk_Height/2 )

        # Set window in center screen with following way.
        self.w.geometry("{}x{}+{}+{}".format(850,600,positionRight, positionDown))
        self.f1 = tk.Frame(width=1150, height=200, background="gray")
        
        self.f1.pack(fill="both", expand=True, padx=40, pady=20)
       
        self.w.title("Result ...")
        self.w['bg']='#D8F2F1'
        self.f1['bg']='#D8F2F1'
        self.UI()

    def UI(self):
        style = ttk.Style()

        style.configure("BW.TLabel", foreground="white", background="gray",font=('Helvetica', 14))
        ttk.Style().configure('pad.Label', padding='5 1 1 1')
        self.l1 = ttk.Label(self.w, text="Enter Age ",style="BW.TLabel")
        self.l1.place(x=50, y=20,w=250,h=30)

        style.configure("TEntry", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        self.s1 = tk.StringVar()
        self.t1 = ttk.Entry(self.w,textvariable=self.s1,style="TEntry")
        self.t1.place(x=320, y=20,w=370,h=30)

        style.configure("BW.TLabel", foreground="white", background="gray",font=('Helvetica', 14))
        ttk.Style().configure('pad.Label', padding='5 1 1 1')
        self.l1 = ttk.Label(self.w, text="Enter Previous History ",style="BW.TLabel")
        self.l1.place(x=50, y=60,w=250,h=30)

        style.configure("TEntry", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        self.s2 = tk.StringVar()
        self.t2 = ttk.Entry(self.w,textvariable=self.s2,style="TEntry")
        self.t2.place(x=320, y=60,w=370,h=30)

        style.configure("BW.TLabel", foreground="white", background="gray",font=('Helvetica', 14))
        ttk.Style().configure('pad.Label', padding='5 1 1 1')
        self.l1 = ttk.Label(self.w, text="Enter Tests prescribed ",style="BW.TLabel")
        self.l1.place(x=50, y=110,w=250,h=30)

        style.configure("TEntry", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        self.s3 = tk.StringVar()
        self.t3 = ttk.Entry(self.w,textvariable=self.s3,style="TEntry")
        self.t3.place(x=320, y=110,w=370,h=30)

        style.configure("TButton", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        b2 = ttk.Button(self.w, text="Result",style="TButton")
        b2.place(x=50, y=150,w=150,h=30)
        b2['command'] = lambda: self.process()

        style.configure("TButton", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        b3 = ttk.Button(self.w, text="End",style="TButton")
        b3.place(x=220, y=150,w=150,h=30)
        b3['command'] = lambda: self.endprocess()        

        style.configure("BW.TLabel", foreground="white", background="gray",font=('Helvetica', 14))
        ttk.Style().configure('pad.Label', padding='5 1 1 1')
        self.strt2=tk.StringVar()
        self.l1 = ttk.Label(self.w, textvariable=self.strt2,style="BW.TLabel",font=('Helvetica', 34))
        self.l1.place(x=50, y=200,w=650,h=230)

    def process(self):
        from DDPG import Test
        t=Test(None)
        input =self.s3.get()#Test
        str2 =self.s1.get()#age
        str1 =self.s2.get()#history
        if input is None or input.strip()=="":
            messagebox.showinfo("Result", "Please enter the Tests Precribed")    
            return
        elif str2 is None or str2.strip()=="":
            messagebox.showinfo("Result", "Please enter the Age")    
            return
        result=t.getResult(input,str2,str1)    
        self.strt2.set("Possible Ward assignment \nwill be "+result)    
        #messagebox.showinfo("Result", "Possible Ward assignment will be "+result)    

    def endprocess(self):
            self.w.withdraw()
            

    



if __name__ == "__main__":
    main()
