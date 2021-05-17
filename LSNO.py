
import numpy as np




__all__ = ['LSNO']

class LSNO(object):
    def __init__(self,problem="Maxq",n=10):
        
        assert type(n)==int, "n should be an integer"
        assert type(problem)==int or type(problem)==str, "Require problem to be name(str) or index(int)"
        
        self.prob_index= -1
        if type(problem)==int:
            self.prob_index=problem
            
        if type(problem)==str:
            if problem=="Maxq":
                self.prob_index=0
            if problem=="Mxhilb":
                self.prob_index=1
            if problem=="Lq":
                self.prob_index=2
            if problem=="Cb31":
                self.prob_index=3
            if problem=="Cb32":
                self.prob_index=4
            if problem=="ActiveFaces":
                self.prob_index=5
            if problem=="Brown":
                self.prob_index=6
            if problem=="Mifflin":
                self.prob_index=7
            if problem=="Crescent1":
                self.prob_index=8
            if problem=="Crescent2":
                self.prob_index=9
        assert self.prob_index<=9 and self.prob_index>=0
            
        self.n=n
        self.h=None
        self.getD=None
        self.F=None
        self.phi=None
        self.phiprime=None
        self.x0=None
        self.L_np=None
        self.L_h=None
        self.m=None
        
    def phi_0(self,x):
        return -np.sum(x[:-1])+4*x.dot(x)-2*x[0]**2-2*x[-1]**2-2*len(x)+2
    def phiprime_0(self,x):
        result=8*x
        result[0]/=2
        result[-1]/=2
        result[:-1]-=1
        return result.reshape(-1,1)
    def phi_1(self,x):
        return 0
    def phiprime_1(self,x):
        return 0
    def phi_2(self,x):
        return -np.exp(-x.dot(x))
    def phiprime_2(self,x):
        return (-self.phi_2(x)*2*x).reshape(-1,1)
        
    def set_phi(self):
        n=self.prob_index
        if n==7:
            self.phi=self.phi_0
            self.phiprime=self.phiprime_0
            self.L_np=100
        elif n==6:
            self.phi=self.phi_2
            self.phiprime=self.phiprime_2
            self.L_np=1
        else:
            self.phi=self.phi_1
            self.phiprime=self.phiprime_1
            self.L_np=0
        return 0
            
    def h_0(self,z):
        return np.max(z)
    def getD_0(self,z):
        index=np.where(z==np.max(z))[0]
        result=np.zeros((len(z),len(index)))
        for i in range(len(index)):
            result[index[i],i]=1
        return result
    
    def h_1(self,z,compare=2):
        assert len(z)%compare==0,"length of z is wrong"
        result=0
        m=len(z)
        for i in range(m//compare):
            result+=np.max([z[i+j*m//compare] for j in range(compare)])
        return result
    def getD_1(self,z,compare=2):
        assert len(z)%compare==0,"length of z is wrong"
        result=np.zeros(z.shape).reshape(-1,1)
        m=len(z)
        for i in range(m//compare):
            local=np.array([z[i+j*m//compare] for j in range(compare)])
            temp=np.where(local==np.max(local))[0]
            if len(temp)==1:
                for k in range(result.shape[1]):
                    result[i+temp*m//compare,k]=1
            else:
                resultsave=result
                width=result.shape[1]
                for _ in range(len(temp)-1):
                    result=np.hstack((result,resultsave))
                for k in range(len(temp)):
                    result[i+temp[k]*m//compare,(k*width):(k*width+width)]=1
        return result
    
    def h_2(self,z,compare=3):
        assert len(z)%compare==0,"length of z is wrong"
        result=0
        m=len(z)
        for i in range(m//compare):
            result+=np.max([z[i+j*m//compare] for j in range(compare)])
        return result
    def getD_2(self,z,compare=3):
        assert len(z)%compare==0,"length of z is wrong"
        result=np.zeros(z.shape).reshape(-1,1)
        m=len(z)
        for i in range(m//compare):
            local=np.array([z[i+j*m//compare] for j in range(compare)])
            temp=np.where(local==np.max(local))[0]
            if len(temp)==1:
                for k in range(result.shape[1]):
                    result[i+temp*m//compare,k]=1
            else:
                resultsave=result
                width=result.shape[1]
                for _ in range(len(temp)-1):
                    result=np.hstack((result,resultsave))
                for k in range(len(temp)):
                    result[i+temp[k]*m//compare,(k*width):(k*width+width)]=1
        return result
    
    def h_3(self,z):
        return np.linalg.norm(1-z**2,ord=1)
    def getD_3(self,z):
        result=np.zeros(z.shape).reshape(-1,1)
        m=len(z)
        for i in range(m):
            if abs(z[i])>1:
                result[i,:]=2*z[i]
            elif abs(z[i])<1:
                result[i,:]=-2*z[i]
            else:
                result=np.hstack((result,result))
                result[i,:(result.shape[1]//2)]=2
                result[i,(result.shape[1]//2):]=-2
        return result
    
    def h_4(self,z):
        return np.linalg.norm(z,ord=1)
    def getD_4(self,z):
        m=len(z)
        # take L1 norm as example
        if (z!=0).all():
            return np.sign(z).reshape(-1,1)
        else:
            D=np.sign(z).reshape(-1,m)
            for i in range(m):
                j=0
                while(j<len(D)):
                    if D[j,i]==0:
                        temp=D[j,:].reshape(-1,m)
                        temp[0,i]+=1
                        D=np.concatenate((D,temp))
                        temp[0,i]-=2
                        D=np.concatenate((D,temp))
                        D = np.delete(D,j, axis = 0).reshape(-1,m)
                    else:
                        j+=1
            return D.T
    def h_5(self,z):
        return np.max(abs(z))
    def getD_5(self,z):
        if (z==0).all():
            return np.zeros(len(z)).reshape(-1,1)
        index=np.where(abs(z)==np.max(abs(z)))[0]
        result=np.zeros((len(z),len(index)))
        for i in range(len(index)):
            result[index[i],i]=np.sign(z[index[i]])
        return result
    
            
        
    def set_h(self):
        n=self.prob_index
        if (n==np.array([0,4,8])).any():
            self.h=self.h_0
            self.getD=self.getD_0
            self.L_h=1
        elif (n==np.array([2,5,9])).any():
            self.h=self.h_1
            self.getD=self.getD_1
            self.L_h=1
        elif n==3:
            self.h=self.h_2
            self.getD=self.getD_2
            self.L_h=1
        elif n==6:
            self.h=self.h_3
            self.getD=self.getD_3
            self.L_h=3
        elif n==7:
            self.h=self.h_4
            self.getD=self.getD_4
            self.L_h=1
        elif n==1:
            self.h=self.h_5
            self.getD=self.getD_5
            self.L_h=1
    
    def F_0(self,x):
        return x**2
    def F_1(self,x):
        result=[]
        for i in range(1,len(x)+1):
            result=np.append(result,np.sum([x[j]/(i+j) for j in range(len(x))]))
        return result
    def F_2(self,x):
        result1=[]
        result2=[]
        for i in range(len(x)-1):
            result1=np.append(result1,-x[i]-x[i+1])
            result2=np.append(result2,-x[i]-x[i+1]+x[i]**2+x[i+1]**2-1)
        return np.append(result1,result2)
    def F_3(self,x):
        result1=[]
        result2=[]
        result3=[]
        for i in range(len(x)-1):
            result1=np.append(result1,x[i]**4+x[i+1]**2)
            result2=np.append(result2,(2-x[i])**2+(2-x[i+1])**2)
            result3=np.append(result3,2*np.exp(-x[i]+x[i+1]))
        return np.concatenate([result1,result2,result3])
    def F_4(self,x):
        result1=0
        result2=0
        result3=0
        for i in range(len(x)-1):
            result1+=x[i]**4+x[i+1]**2
            result2+=(2-x[i])**2+(2-x[i+1])**2
            result3+=2*np.exp(-x[i]+x[i+1])
        return np.array([result1,result2,result3])
    def F_5(self,x):
        result1=[]
        result2=[]
        for i in range(len(x)):
            result1=np.append(result1,np.log(abs(np.sum(x))+1))
            result2=np.append(result2,np.log(abs(x[i])+1))
        return np.append(result1,result2)
    
    def F_6(self,x):
        assert len(x)==2,"here"
        return np.array([(x[1]-x[0]**2)**2,(1-x[0])**2])
    
    def F_7(self,x):
        result=[]
        for i in range(len(x)-1):
            result=np.append(result,1.75*(x[i]**2+x[i+1]**2-1))
        return result
    def F_8(self,x):
        result1=0
        result2=0
        for i in range(len(x)-1):
            result1+=x[i]**2+(1-x[i+1])**2+x[i+1]-1
            result2+= -x[i]**2-(1-x[i+1])**2+x[i+1]+1
        return np.array([result1,result2])
    def F_9(self,x):
        result1=[]
        result2=[]
        for i in range(len(x)-1):
            result1=np.append(result1,x[i]**2+(1-x[i+1])**2+x[i+1]-1)
            result2=np.append(result2,-x[i]**2-(1-x[i+1])**2+x[i+1]+1)
        return np.append(result1,result2)
    
    def set_F(self):
        n=self.prob_index
        if n==0:
            self.F=self.F_0
            self.m=self.n
            
        elif n==1:
            self.F=self.F_1
            self.m=self.n
        elif n==2:
            self.F=self.F_2
            self.m=2*self.n-2
        elif n==3:
            self.F=self.F_3
            self.m=3*self.n-3
        elif n==4:
            self.F=self.F_4
            self.m=3
        elif n==5:
            self.F=self.F_5
            self.m=2*self.n
        elif n==6:
            self.F=self.F_6
            self.m=2
        elif n==7:
            self.F=self.F_7
            self.m=self.n-1
        elif n==8:
            self.F=self.F_8
            self.m=2
        elif n==9:
            self.F=self.F_9
            self.m=2*self.n-2
        
        
    def set_x0(self):
        x0=np.zeros(self.n)
        n=self.prob_index
        if n==0:
            for i in range(self.n):
                if i<self.n/2:
                    x0[i]=i
                else:
                    x0[i]=-i
        if n==1:
            x0=x0+1
        if n==2:
            x0=x0-0.5
        if n==3:
            x0=x0+2
        if n==4:
            x0=x0+2
        if n==5:
            x0=x0+1
        if n==6:
            x0=np.random.random(2)
        if n==7:
            x0=x0-1
        if n==8:
            for i in range(self.n):
                if i%2==1:
                    x0[i]=-1.5
                else:
                    x0[i]=2
        if n==9:
            for i in range(self.n):
                if i%2==1:
                    x0[i]=-1.5
                else:
                    x0[i]=2
        self.x0=x0
        
    def setup(self):
        self.set_phi()
        self.set_F()
        self.set_h()
        self.set_x0()
    
    
    def get_prob(self):
        return self.h,self.getD,self.F,self.phi,self.phiprime,\
            self.x0,self.L_np,self.L_h,self.m
        
        
        
        
        
    
    
        