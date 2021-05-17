import numpy as np
import time
from cvxopt import matrix, solvers
import sys
sys.path.append("./ComponentBuilding")
import model

__all__ = ['MS']



class MS(object):
    def __init__(self,problem,nsample=10):

        self.n=problem.n
        self.m=problem.m
        self.h=problem.h
        self.get_D=problem.getD
        self.F=problem.F
        self.phi=problem.phi
        self.phiprime=problem.phiprime
        self.xsave=np.array([problem.x0]).reshape(-1,1)
        self.x=problem.x0
        self.L_np=problem.L_np
        self.L_h=problem.L_h
        self.nsample=int(max(nsample,round((1+1/np.log(self.n))*self.n)))
        self.fsave=np.array([self.f()])
        self.delta_max=1e4
        self.delta_min=1e-8
        self.delta=self.delta_max/10
        self.ifprint=1
        self.eta_1=0.1
        self.eta_2=1
        self.kappa_d=1e-4
        self.kappa_H=1
        self.gamma_d=0.5
        self.gamma_i=1.5
        self.conv_tol=1e-9
        self.common_flag=1
        self.save_fvalue=1e5
        self.searchstep=5
        self.max_iter=3000
        self.iter=0
        self.time_start=time.time()
        self.time=np.array([time.time()])
        self.time_end=None
    
    def f(self,x="None"):
        if x=="None":
            return self.phi(self.x)+self.h(self.F(self.x))
        else:
            return self.phi(x)+self.h(self.F(x))
    
    
    
    def unit_ball(self,N=1):
        result=np.zeros([N,self.n])
        denorminator=np.random.uniform(size=N)
        for i in range(N):
            numerator=np.random.normal(size=self.n)
            result[i]=numerator/np.sqrt(numerator.dot(numerator))*pow(denorminator[i],1/self.n)
        if N==1:
            return result.ravel()
        else:
            return result.T
        

    
    def grid_search(self,s):
        Fx=self.F(self.x)
        Fxs=self.F(self.x+s)
        diff=(Fx-Fxs)
        fx=self.f()
        fxs=self.f(self.x+s)
        epsilon=1e-5*fx
        threshold=fx-fxs+epsilon
        
        
        if (np.matmul(diff,self.get_D(Fx))-s.dot(self.phiprime(self.x))<=threshold).any():
            idx=np.where(np.matmul(diff,self.get_D(Fx))-s.dot(self.phiprime(self.x))<=threshold)[0][0]
            return Fx,self.get_D(Fx)[:,idx].reshape(-1,1)
        elif (np.matmul(diff,self.get_D(Fxs))-s.dot(self.phiprime(self.x+s))<=threshold).any():
            idx=np.where(np.matmul(diff,self.get_D(Fxs))-s.dot(self.phiprime(self.x+s))<=threshold)[0][0]
            return Fxs,self.get_D(Fxs)[:,idx].reshape(-1,1)
        for l in range(19):
            assert l<=18 
            candidates=np.array([(2*2**k-1)/(2**l) for k in range(l)])
            for k in range(l):
                alpha=candidates[k]
                z=alpha*Fx+(1-alpha)*Fxs
                #print("gradiant",get_D(z).T,Fx,Fxs)
                #print("diff",np.matmul(diff,get_D(z)),h(Fx)-h(Fxs))
                if (np.matmul(diff,self.get_D(z))-s.dot(self.phiprime(self.x+(1-alpha)*s))<=threshold).any():
                    idx=np.where(np.matmul(diff,self.get_D(z))-s.dot(self.phiprime(self.x+(1-alpha)*s))<=threshold)[0][0]
                    return z,self.get_D(z)[:,idx].reshape(-1,1)
        return Fxs,self.get_D(Fxs)[:,0].reshape(-1,1)

    def m_F(self):
        radius=min(self.delta,self.delta_max/1e6)
        xl=np.array([-1e10]*self.n)
        xu=np.array([1e10]*self.n)
        M_F=model.Model(npt=self.nsample,x0=self.x,r0=self.F(self.x),xl=xl,xu=xu,r0_nsamples=1)
        for i in range(1,self.nsample):
            noise=self.unit_ball()*radius
            M_F.change_point(i,noise,self.F(noise+self.x))
       
        M_F.interpolate_mini_models_svd()
        return M_F


    def sub_optimize(self,G):
        m=G.shape[1]
        P = matrix(np.matmul(G.T,G))
        q = matrix(np.zeros(m))
        Go = matrix(-np.eye(m))
        h = matrix(np.zeros(m))
        A = matrix(np.ones(m)).T
        b = matrix(np.ones(1))
        result = solvers.qp(P,q,Go,h,A,b,kktsolver='ldl',options={'kktreg':1e-9,"show_progress":False})
        
        return np.array(result["x"])

    def M(self,x,M_F):
        d=x-M_F.xbase
        return M_F.model_value(d,d_based_at_xopt=False,with_const_term=True).reshape(M_F.m(),1)

    def Optimize(self): 
        
        Z=[]
        G=[]
        D=[]
        g=[]
        d=[]
        num=1
        den=1e5
        count=0
        while(self.iter<self.max_iter and self.common_flag):
            self.iter+=1
            timetemp=time.time()
            self.time=np.append(self.time,timetemp-self.time_start)
            self.xsave=np.hstack((self.xsave,self.x.reshape(-1,1)))
            #explosion
            if np.linalg.norm(self.x)>1e4:
                self.common_flag=0
            self.fsave=np.append(self.fsave,self.f())
            if self.ifprint:
                print(self.iter,self.f(),self.delta)
            #Build p dimension component model m^F
            M_F=self.m_F()
            
            jac_F=M_F.get_final_results()[3]
            #Get Z and D in two ways
            
            Xs=self.unit_ball(N=self.nsample)*self.delta+np.tile(self.x.reshape(-1,1),self.nsample)
            Z=np.array([self.F(Xs[:,i]) for i in range(self.nsample)]).T
            D=np.array([self.get_D(Z[:,i]).ravel() for i in range(self.nsample)]).T
            #first part
    
            while(self.common_flag):
                ##get derivative of m^F
                M_F.interpolate_mini_models_svd()
                jac_F=M_F.get_final_results()[3]
    
                ##get G by m^F' and D
                G=np.matmul(jac_F.T,D)+np.tile(self.phiprime(self.x),D.shape[1])
                ##solve quadratic optimization to get d and ultimately g
                lamda=self.sub_optimize(G/np.linalg.norm(G))
                g=np.matmul(G,lamda)
                #print("G",G)
                
                d=np.matmul(D,lamda)
                g_norm=np.linalg.norm(x=g, ord=2)
                if (self.delta<self.delta_min):
                    self.common_flag=0
                    break
                if (self.delta<self.eta_2*g_norm):
                    s=np.ravel(-self.delta*g/g_norm)
                    ## small step and check sufficient decrease
    
                    terminating_const=self.kappa_d*g_norm/2*min(self.delta,np.sqrt(self.n)*g_norm/(self.L_h*self.kappa_H+self.L_np))
                    for i in range(1,self.searchstep):
                        s=i/self.searchstep*s/np.linalg.norm(s,ord=2)*self.delta
                        #print("checkdecrease",M(x,M_F),M(x+s,M_F),d,phi(x)-phi(x+s),terminating_const)
                        if np.matmul((self.M(self.x,M_F)-self.M(self.x+s,M_F)).T,d).ravel()+self.phi(self.x)-self.phi(self.x+s)>=terminating_const:
                            break
                    ###generate s (next iter points) and (j,z)
                    z,tilda_h=self.grid_search(s)
                    flag=0
                    #print(Z.shape,z.reshape(-1,1).shape)
                    Z=np.hstack((Z,z.reshape(-1,1)))
                    
                    ######################
                    for i in range(Z.shape[1]):
                        
                        if (self.get_D(Z[:,i].reshape(-1,1))==tilda_h).all():
                            temp=np.matmul(s,jac_F.T)
                            temp=np.matmul(temp,tilda_h-d)
                            if (temp<=1e-10).all():
                                flag=1
                                break
                            
                    if flag==0:
                        #print("here?")
                        s=s/np.linalg.norm(s,ord=2)*self.delta
                        z,tilda_h=self.grid_search(s)
                        break
    
                    deriv_z=self.get_D(z.reshape(-1,1))
                    num_deriv_z=deriv_z.shape[1]
                    search=np.array([(deriv_z[:,i].reshape(-1,1)==tilda_h).all() for i in range(num_deriv_z)])
                    if search.any():
                        #### get rho
                        num=np.matmul((self.F(self.x)-self.F(self.x+s)).T,d).ravel()+self.phi(self.x)-self.phi(self.x+s)
                        den=np.matmul((self.M(self.x,M_F)-self.M(self.x+s,M_F)).T,d).ravel()+self.phi(self.x)-self.phi(self.x+s)
                        #rho=num/den
                        break
                    else:
                        x_new=self.unit_ball()
                        M_F.add_new_point(x_new,self.F(x_new+self.x))
                        Z=self.F(M_F.points.T+np.tile(self.x,M_F.num_pts).reshape(-1,M_F.num_pts).T)
                        Z=np.hstack((Z,z.reshape(-1,1)))
                        D=np.array([self.get_D(Z[:,i]).ravel() for i in range(Z.shape[1])]).T
                        #### update Z
                else:
                    num=0
                    break
            # second part
            #print("")
            #print(x,s)
            #print(num/den,self.eta_1)
            if num>=self.eta_1*den:
                
                #print(h(func(x))+phi(x),h(func(x+s))+phi(x+s))
                self.x=self.x+s
                if abs(self.save_fvalue-self.f())<self.conv_tol:
                    count+=1
                    if count==5:
                        count=0
                        break
                else:
                    self.save_fvalue=self.f()
                
                self.delta=min(self.gamma_i*self.delta,self.delta_max)
                #print(delta,"delta")
            else:
                self.delta=self.gamma_d*self.delta
                #print(delta,"heredelta")
        
          
        print(self.x,self.f(),self.delta)
        self.time_end=time.time()
        return self.time_end-self.time_start,self.f(),self.x
    
    
    

        
                