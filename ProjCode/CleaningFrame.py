import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
import globaldata
from tkinter.scrolledtext import ScrolledText
import re,os
from collections import defaultdict
from BatchFrame import BatchFrame
#pip install scikit-learn

from sklearn import preprocessing

folderpath=""

def main():
   
    program = CleaningFrame()
    program.w.mainloop()
    
class CleaningFrame:
    def __init__(self, root):
        self.w = root
        self.w = root
        Tk_Width = 1250
        Tk_Height = 550
 
         #calculate coordination of screen and window form
        positionRight = int( self.w.winfo_screenwidth()/2 - Tk_Width/2 )
        positionDown = int( self.w.winfo_screenheight()/2 - Tk_Height/2 )

        # Set window in center screen with following way.
        self.w.geometry("{}x{}+{}+{}".format(1150,600,positionRight, positionDown))
        self.f1 = tk.Frame(width=1150, height=200, background="gray")
        
        self.f1.pack(fill="both", expand=True, padx=40, pady=20)
       
        self.w.title("Cleaning...")
        self.w['bg']='#D8F2F1'
        self.f1['bg']='#D8F2F1'
        self.UI()

    def UI(self):
        style = ttk.Style()

        

        style.configure("TButton", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        b2 = ttk.Button(self.w, text="Preprocess",style="TButton")
        b2.place(x=50, y=20,w=150,h=30)
        b2['command'] = lambda: self.process()

        style.configure("TButton", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        b3 = ttk.Button(self.w, text="Next",style="TButton")
        b3.place(x=240, y=20,w=150,h=30)
        b3['command'] = lambda: self.open_window(BatchFrame)        

        style.configure("Treeview", foreground="black", background="white",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        self.treeframe=ttk.Frame(self.w)
        self.treeframe.place(x=50, y=110,w=720,h=400)
        self.vsb = ttk.Scrollbar(self.treeframe, orient="vertical")
        self.tree = ttk.Treeview(self.treeframe, column=('Gender','Age','Previous_History','Tests','Wards'), show='headings',yscrollcommand=self.vsb.set)
        self.tree.pack(side=tk.LEFT,fill=tk.Y)
        self.vsb.configure(command=self.tree.yview)
        self.tree.column("# 1", width=100, anchor=tk.CENTER)
        self.tree.heading("# 1", text="Gender")

        self.tree.column("# 2", width=100, anchor=tk.CENTER)
        self.tree.heading("# 2", text="Age")

        self.tree.column("# 3", width=100, anchor=tk.CENTER)
        self.tree.heading("# 3", text="Previous_History")

        self.tree.column("# 4", width=200, anchor=tk.CENTER)
        self.tree.heading("# 4", text="Tests") 

        self.tree.column("# 5", width=200, anchor=tk.CENTER)
        self.tree.heading("# 5", text="Wards") 

        style.configure("BW.TLabel", foreground="white", background="gray",font=('Helvetica', 14))
        ttk.Style().configure('pad.Label', padding='5 1 1 1')
        self.lbl1 = ttk.Label(self.w, text="click on preprocess button to\n perform mean value\nsubstitution for missing value\nand data is encoded using\nlabel encoder ",style="BW.TLabel")
        self.lbl1.place(x=800, y=110,w=250,h=120)
    

    def process(self):
        globaldata.DF_admission_record['Age'] = globaldata.DF_admission_record['Age'].fillna(globaldata.DF_admission_record['Age'].mean())
        
        
        
        globaldata.DF_admission_record=globaldata.DF_admission_record.apply(preprocessing.LabelEncoder().fit_transform)

        globaldata.DF_admission_record.to_csv("e:\\test.csv")
        columns = ["gender","Age","Previous_History","Tests","Wards"]
        for i in globaldata.DF_admission_record.index :
            if i>10000:
                break
            gender=globaldata.DF_admission_record['gender'][i]
            Age=globaldata.DF_admission_record['Age'][i]
            Previous_History=globaldata.DF_admission_record['Previous_History'][i]
            Tests=globaldata.DF_admission_record['Tests'][i]
            Diagnose=globaldata.DF_admission_record['Wards'][i]
            t=(gender,Age,Previous_History,Tests,Diagnose)
            print(t)
            self.tree.insert('', 'end', text="1", values=t)


    def open_window(self,Win_class):
            self.w.withdraw()
            global win
            win=tk.Toplevel(self.w)
            Win_class(win)

    



if __name__ == "__main__":
    main()
