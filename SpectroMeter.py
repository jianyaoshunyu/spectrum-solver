import pint 
import scipy.constants as C
from scipy.signal import argrelmax
from scipy.ndimage import gaussian_filter

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LogNorm
from tkinter import filedialog, messagebox

import os
ureg=pint.UnitRegistry()

ee,cc,me=C.elementary_charge,C.speed_of_light,C.electron_mass
dt=1e-4/cc
resolution=1000 #1 meter /1000
try:
    Bnew=np.load('Bnew.npy')
except FileNotFoundError:
    print('Magnet not Found')

def getB(x,y):#meter
    i=int(y*resolution)
    j=int(x*resolution)
    if 0<=i<0.23*resolution and 0<=j<0.198*resolution:
        return np.abs(Bnew[i,j]/10) #from kgauss to T
    else:
        return 0

def dataAccess(metadir):
    datafile=[]
    for root,dirs,files in os.walk(metadir):
        for file in files:
            if file.endswith('.asc'):
                pre=os.path.splitext(file)[0]
                new_name=pre+'.npy'
                if new_name not in files:
                    d=np.genfromtxt(file)
                    data=np.delete(d,0,axis=1)
                    new_path=os.path.join(root,new_name)
                    np.save(new_path,data)
                    datafile.append(new_name)
    print('generated:',datafile)

class hit:
    num=0
    def __init__(self,x, energy):
      self.energy = energy
      # self.inpixel = i
      self.x = x #where the e hits the outCCD
      print()
      hit.num+=1

    def __lt__(self,other):
      return self.x<other.x
    def __gt__(self,other):
      return self.x>other.x

    @staticmethod
    def count():
      return hit.num
      
class Electron:
    def __init__(self,energy,inx):
      self.x=inx#(pixel)/100
      self.y,self.t=0,0
      self.gamma=energy*1e6*ee/(me*cc*cc)
      self.beta=np.sqrt(1-1/(self.gamma*self.gamma))
      self.ang=0#divergence(pixel)
      self.v=self.beta*cc
      self.vy= self.v*np.cos(self.ang)
      self.vx= self.v*np.sin(self.ang)
      self.m=self.gamma*me
      self.energy=energy
      # self.pixel =pixel

    def show(self):
      print('x:',self.x)
      print('y:',self.y)
      print('vx:',self.vx)
      print('vy:',self.vy)
      # print('ang:',self.ang)
      print('gamma',self.gamma)
    
    def onestep(self):
      B=getB(self.x,self.y)
      #r=self.m*self.v/np.abs(ee*B)
      omega=np.abs(ee*B/self.m)
      #sin,cos=self.vx/self.v,self.vy/self.v
      #self.x+=r*(1-np.cos(omega*dt))*sin
      self.x+=self.vx*dt
      self.y+=self.vy*dt
      self.ang+=omega*dt
      self.vx=self.v*np.sin(self.ang)
      self.vy=self.v*np.cos(self.ang)
       
    def run(self):
      #fig,ax=plt.subplots()
      #x0=[]
      #y0=[]
      i=0
      while(i<5000):
        #x0.append(self.x)
        #y0.append(self.y)
        x1,y1=self.x,self.y
        self.onestep()
        if self.y>=0.23 and 0.0245<=self.x<=0.19:
          finalx=x1+(self.x-x1)*(0.23-y1)/(self.y-y1)
          #ax.scatter(x0,y0)
          return hit(finalx,self.energy)
        elif self.x<0:
          #print('hit right')
          return 1
        elif self.x>0.198:
          #print('hit left')
          return 3
        elif self.y<0:
          #print('hit bottom')
          return 4
        elif self.y>0.23:
          #print('run away')
          return 2
        i+=1
      #ax.scatter(x0,y0)
      return 0

class Spectrum:
    def __init__(self,num):
        self.shot=num
        self.inpoint=self.getEnterpos()#354
        self.e=self.getEnergy()#self.E1(self.inpoint,np.arange(243,985))#energy along pixel
        self.x=self.getDensity()# number density along pixel 
        self.p=self.E1(354,np.where(self.x==np.max(self.x))[0][0]+243)# peak energy

    def outx(self,i):#position of pixel i at exit
        return (i-85)*15.5/910+2.45 #cm from pixel 86
    def inx(self,i):#position of pixel i at entrance
        return (i-278)*8.8/214+2 #cm from pixel 279
    
    def E1(self,i,j):
        y=23#magnet shape in centimeter 
        delta=self.outx(j)-self.inx(i)
        B0=-0.8
        r=(y*y+delta*delta)/(2*delta)# centimeter
        R=r/100#meter
        E=np.sqrt(R*R*ee*ee*B0*B0*cc*cc+me**2*cc*4)# IS
        return ((E*ureg.J).to('MeV')).magnitude #MeV

    def simulate(self):
        target=str(self.shot)+'hits'+'.npy'
        if((target) in os.listdir('./Data/hits/')):
            h=np.load('./Data/hits/'+target,allow_pickle=True)#history problem
            return h
        else:
            temp=[]
            for e in np.arange(71,500):
                particle=Electron(e,self.inx(self.inpoint)/100).run()
                if isinstance(particle,hit):
                    temp.append(particle)
            temp.sort()
            h=np.array(temp)
            np.save('./Data/hits/'+target,h)
            return h
    
    def getEnergy(self):
        hits=self.simulate()
        e=[]
        for i in np.arange(243,985):
            t=hit(self.outx(i)/100,0)
            idx=np.searchsorted(hits,t)
            energy=0
            if(idx>=len(hits)): 
                energy=70
            elif(idx-1<0):
                energy=hits[0].energy
            else:
                energy=(hits[idx-1].energy+hits[idx].energy)/2
            e.append(energy)
        return e    
    
    def getDensity(self):
        data=np.load('./Data/exist/'+str(self.shot)+'.npy')
        data=data[:,400:800]
        x=np.sum(data,axis=1)[243:985]
        x0=x-x.min()
        x1=x0/np.sum(x0)
        return x1

    def getEnterpos(self):
        data=np.load('./Data/entry/'+str(self.shot)+'.npy')
        smoothed_data = gaussian_filter(data, sigma=12, truncate=4)

        # Find the indices of all the local maxima in the flattened data array
        max_indices = argrelmax(smoothed_data.flatten())[0]#[0] makes tuple become list

        # Convert the flattened indices back to 2D indices
        max_indices_2d = np.unravel_index(max_indices, data.shape)

        # Find the index of the maximum value among the local maxima
        peak_index = np.argmax(smoothed_data[max_indices_2d])

        # Get the peak value and its indices
        peak_value = smoothed_data[max_indices_2d][peak_index]
        peak_indices = (max_indices_2d[0][peak_index], max_indices_2d[1][peak_index])

        return peak_indices[0]

class SpectrumPlotter:
    def __init__(self, master):
        self.buffer=[]

        self.master = master
        self.master.title("Spectrum Meter")
        self.master.geometry("1000x650")

        # Create four Entry widgets for inputting parameters
        self.param_frame = tk.Frame(self.master)
        self.param_frame.place(x=20,y=20)#pack(side="left", padx=20, pady=20)
        self.param_entries = []
        for i in range(4):
            if(i==0):
                label = tk.Label(self.param_frame, text=f"Main Shot")
            else:
                label = tk.Label(self.param_frame, text=f"Contrast Shot{i}")
            label.pack()
            entry = tk.Entry(self.param_frame, width=10)
            entry.insert(0, str(29+i))
            entry.pack()
            self.param_entries.append(entry)

        # Create four Canvas widgets for displaying plots
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.place(x=150,y=20)#pack(side="right", padx=20, pady=20)
        self.figures = []
        for i in range(4):
            fig, ax = plt.subplots()
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.get_tk_widget().grid(row=i//2, column=i%2)
            self.figures.append(fig)

        # Create a button for plotting
        plot_button = tk.Button(self.master,width=10,text="Plot", command=self.plot)
        plot_button.place(x=20,y=200)#pack(side="bottom", pady=20)

        plot_button2 = tk.Button(self.master,width=10,text="Save", command=self.save)
        plot_button2.place(x=20,y=250)#pack(side="bottom", pady=20)

        self.label = tk.Label(master, text="")
        self.label.place(x=20,y=300)

    def plot(self):
        # Get input parameters
        params = []
        for entry in self.param_entries:
            try:
                param = int(entry.get())
            except ValueError:
                param = 1  # default value
            params.append(param)
        
        self.buffer.clear()

        dirofentry='./Data/entry/'
        dirofexist='./Data/exist/'
      
      
        for i, fig in enumerate(self.figures):
            ax = fig.gca()
            ax.cla()
            # x = np.linspace(-np.pi, np.pi, 100)
            # y = np.sin(params[i] * x)
            # ax.plot(x, y)
            if(i==0):
                # print(dirofentry+str(params[i])+'.npy')
                try:
                    data=np.load(dirofentry+str(params[0])+'.npy')
                except FileNotFoundError:
                    messagebox.showerror("File Not Found", f"The entrydata {str(params[0])+'.npy'} does not exist.")
                ax.imshow(data,norm=LogNorm(),cmap='plasma')
                ax.set_title(f"Image of Entry for Shot{params[0]}")
            elif(i==1):
                # print(dirofexist+str(params[i])+'.npy')
                # with np.load(dirofexist+str(params[i])+'.npy') as data:
                try:
                    data=np.load(dirofexist+str(params[0])+'.npy')
                except FileNotFoundError:
                    messagebox.showerror("File Not Found", f"The existdata {str(params[0])+'.npy'} does not exist.")
                ax.imshow(np.rot90(data),cmap='coolwarm')
                ax.set_title(f"Image of Exist for Shot{params[0]}")
            elif(i==2):
                A=Spectrum(params[0])
                self.buffer.append(A.e)
                self.buffer.append(A.x)
                ax.plot(A.e, A.x)
                ax.set_xlim(70,500)
                ax.set_title('Spectrum',fontsize=10)
                ax.set_xlabel('Energy(MeV)',fontsize=10)
                ax.set_ylabel('Probability Density',fontsize=10)
                ax.axvline(A.p,ls='--',alpha=0.5)
                ax.axhline(np.max(A.x)/2,ls='--',alpha=0.5)
                fig.tight_layout()
            else:
                x = np.linspace(-np.pi, np.pi, 100)
                y = np.sin(params[i] * x)
                ax.plot(x, y)
                ax.set_title(f"Sin Function with Parameter {i+1}")
            fig.canvas.draw()

    def save(self):
        params = []
        for entry in self.param_entries:
            try:
                param = int(entry.get())
            except ValueError:
                param = 1  # default value
            params.append(param)

        dirofsave='./Data/saved/'
        
        self.figures[2].savefig(f'./Data/saved/Single plot of Shot{params[0]}.png')
        np.savetxt(f'./Data/saved/Spectrum of Shot{params[0]}',self.buffer)

        self.label.config(text="Data Saved")

if __name__=='__main__':
    dataAccess('./Data')
    root = tk.Tk()
    app = SpectrumPlotter(root)
    root.mainloop()
