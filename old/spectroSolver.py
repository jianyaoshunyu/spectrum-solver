import numpy as np
import os                                                                                                                                   
import matplotlib.pyplot as plt
plt.style.use(['classic']) 
import scipy.constants as C

def dataAccess():
    datafile=[]
    appendix='.asc'
    for filename in os.listdir('/public1/home/sca1437/zhang/experiment2204/row'):
        pre=os.path.splitext(filename)[0]
        tail=os.path.splitext(filename)[1]
        if tail==appendix:
            d=np.genfromtxt('row/'+filename)
            data=np.delete(d,0,axis=1)
            np.save('processed/'+pre+'.npy',data)
            datafile.append(pre+'.npy')
    print('generated:',datafile)
    return datafile

datafile=dataAccess()

class hit:
    num=0
    def __init__(self,x, energy, i):
      self.energy = energy
      self.inpixel = i
      self.x = x
      hit.num+=1

    def __lt__(self,other):
      return self.x<other.x
    def __gt__(self,other):
      return self.x>other.x

    @staticmethod
    def count():
      return hit.num
      
class Electron:
    def __init__(self,energy,pixel):
      self.x=inx(pixel)/100
      self.y,self.t=0,0
      self.gamma=energy*1e6*ee/(me*cc*cc)
      self.beta=np.sqrt(1-1/(self.gamma*self.gamma))
      self.ang=divergence(pixel)
      self.v=self.beta*cc
      self.vy= self.v*np.cos(self.ang)
      self.vx= self.v*np.sin(self.ang)
      self.m=self.gamma*me
      self.energy=energy
      self.pixel =pixel

    def show(self):
      print('x:',self.x)
      print('y:',self.y)
      print('vx:',self.vx)
      print('vy:',self.vy)
      print('ang:',self.ang)
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
          return hit(finalx,self.energy,self.pixel)
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

for file in datafile:
    pre=os.path.splitext(file)[0]
    data=np.load('processed/'+file)
    x=np.sum(data,axis=1)[85:985]
    x0=x-x.min()
    x1=x0/np.sum(x0)
    fig,ax=plt.subplots()
    ax.plot(np.arange(85,985),x1)
    fig.savefig(pre+'.jpg',dpi=600)
