import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from DataSelFrame import DataSelection

def main():

    
    program = AdminLog()
    program.w.mainloop()
    
class AdminLog:
    def __init__(self):
        self.w = tk.Tk()
        Tk_Width = 350
        Tk_Height = 350
 
         #calculate coordination of screen and window form
        positionRight = int( self.w.winfo_screenwidth()/2 - Tk_Width/2 )
        positionDown = int( self.w.winfo_screenheight()/2 - Tk_Height/2 )

        # Set window in center screen with following way.
        self.w.geometry("{}x{}+{}+{}".format(500,200,positionRight, positionDown))
        self.f1 = tk.Frame(width=200, height=200, background="gray")
        
        self.f1.pack(fill="both", expand=True, padx=40, pady=20)
       
        self.w.title("Sign In...")
        self.w['bg']='#D8F2F1'
        self.f1['bg']='#D8F2F1'
        self.UI()

    def UI(self):
        style = ttk.Style()

        style.configure("BW.TLabel", foreground="white", background="gray",font=('Helvetica', 14))
        ttk.Style().configure('pad.Label', padding='5 1 1 1')
        self.l1 = ttk.Label(self.f1, text="User Name ",style="BW.TLabel")
        self.l1.place(x=50, y=20,w=120,h=30)

        style.configure("TEntry", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        self.strt1=tk.StringVar()
        self.t1 = ttk.Entry(self.f1,textvariable=self.strt1,style="TEntry")
        self.t1.place(x=190, y=20,w=170,h=30)

        style.configure("BW.TLabel", foreground="white", background="gray",font=('Helvetica', 14))
        ttk.Style().configure('pad.Label', padding='5 1 1 1')
        self.l2 = ttk.Label(self.f1, text="Password  ",style="BW.TLabel")
        self.l2.place(x=50, y=60,w=120,h=30)

        style.configure("TEntry", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        self.strt2=tk.StringVar()
        self.t2 = ttk.Entry(self.f1,textvariable=self.strt2,style="TEntry", show="*")
        self.t2.place(x=190, y=60,w=170,h=30)

        style.configure("TButton", foreground="black", background="gray",font = ('courier', 15, 'bold'))
        ttk.Style().configure('pad.Entry', padding='5 1 1 1')
        b1 = ttk.Button(self.f1, text="Submit",style="TButton")
        b1.place(x=50, y=100,w=310,h=40)
        
        b1['command'] = lambda: self.show(DataSelection)

        

    def show(self,Win_class):
        unm=self.strt1.get()
        pwd=self.strt2.get()
        if unm=="admin" and pwd=="1234":
            self.w.withdraw()
            global win
            win=tk.Toplevel(self.w)
            Win_class(win)
        else:
            messagebox.showinfo("Admin Login", "Invalid UserName or Password")    
        
    
if __name__ == "__main__":
    main()
