import sympy
from sympy.physics.mechanics import dynamicsymbols
import numpy as np
import math
import sympy
from sympy import *
from sympy.physics.mechanics import dynamicsymbols
import numpy as np
from scipy.integrate import Radau
from scipy.integrate import odeint

#from controllers import controller
#from controllers import controller
#from . import controller

class CSTRsys():
    def __init__(self):
        s=sympy.Symbol('s')
        t=sympy.Symbol('t')
        
        V=sympy.Symbol('V',real=True)
        k0=sympy.Symbol('k_0',real=True)
        E=sympy.Symbol('E',real=True)
        R=sympy.Symbol('R',real=True)
        Hr=sympy.Symbol('H_r',real=True)
        rho=sympy.Symbol('rho',real=True)
        Cp=sympy.Symbol('C_p',real=True)
        Vp=sympy.Symbol('V_rho',real=True)
        U=sympy.Symbol('U',real=True)
        A=sympy.Symbol('A',real=True)
        Vc=sympy.Symbol('V_c',real=True)
        rhoc=sympy.Symbol('rho_c',real=True)
        Cpc=sympy.Symbol('C_pc',real=True)
        L=sympy.Symbol('L',real=True)
        At=sympy.Symbol('A_t',real=True)
        tauT=sympy.Symbol('tau_T',real=True)
        alp=sympy.Symbol('alpha',real=True)
        Fcmax=sympy.Symbol('F_cmax',real=True)
        F=sympy.Symbol('F')
        Cai=sympy.Symbol('C_ai')
        Tci=sympy.Symbol('T_ci')
        Ti=sympy.Symbol('T_i')
        
        Ca=dynamicsymbols('C_A')
        k=dynamicsymbols('k')
        T=dynamicsymbols('T')
        Tc=dynamicsymbols('T_c')
        m=dynamicsymbols('m')
        #Fc=dynamicsymbols('F_c')
        TO=dynamicsymbols('TO')
        self.ssc={
                    V:7.08,
                    k0:0.0744,
                    E:1.182e7,
                    R:8314.39,
                    Hr:-9.6e7,
                    rho:19.2,
                    Cp:1.815e5,
                    U:3550,
                    A:5.4,
                    Vc:1.82,
                    rhoc:1000,
                    Cpc:4184,
                    L:50.3,
                    At:0.018636,
                    tauT:0.33/60,
                    alp:50,
                     Fcmax:1.2,
                    #Fcmax:1.2*60,
                    F:0.45*60,
                    Cai:2.88,
                    Tci:27,
                    #Ti:66,
                    Ti:88,
        }
        
        #State variables
        self.Xsv=np.array([[Tc],[Ca],[T],[TO]])
        self.U=np.array([[m]])
        
        #Expresiones auxliares
        k=k0*sympy.exp(-(E/(R*(T+273.0))))
        Fc=Fcmax*(alp**(-m))
        
        #eq(13)
        self.eq13= ( (F/V)*(Cai-Ca) - k*((Ca)**2))
        
        #eq(14)
        self.eq14=( F*(Ti-T)/V - k*(Ca**2)*Hr/(rho*Cp) - ((U*A)/(V*rho*Cp))*(T-Tc) )
        
        #eq(15)
        self.eq15=( ((U*A)/(Vc*rhoc*Cpc))*(T-Tc) - Fc*(Tc-Tci)/Vc )
        
        t0=(L*At*rho)/F
        T1=T*(t-t0)
        
        #eq(19)
        self.eq19=(1/tauT)*((T-88.07435096)/(88.07642124-88.07435096)-TO)

        self.sys=np.array([[self.eq15.subs(self.ssc)],
              [self.eq13.subs(self.ssc)],
              [self.eq14.subs(self.ssc)],
              [self.eq19.subs(self.ssc)],
              ])

        self.sisys=lambdify([Tc,Ca,T,TO,m],self.sys)

        self.initial_conditions(z0=[50.5,2.113,88,0])
        self.set_states_mean(Xm=self.z0,Um=0)

    def initial_conditions(self,z0=[50.5,2.113,88,0]):
        self.z0=z0

    def sym_lin_sys(self):
            self.slA=(sympy.Matrix(self.sys).jacobian(self.Xsv))
            self.slB=(sympy.Matrix(self.sys).jacobian(self.U))

    def lin_sys(self,Xc,Uc):
        Vtr=list(np.vstack((self.Xsv,self.U)).T[0])
        lA=self.slA.subs(dict(zip(Vtr,list(np.hstack((Xc,Uc))))))
        lB=self.slB.subs(dict(zip(Vtr,np.hstack((Xc,Uc)))))
        return lA,lB
    
    def set_states_mean(self,Xm,Um):
        self.Xm=Xm
        self.Um=Um
    
    def lmodel(self,X,t,u):
        A,B=self.lin_sys(X,u)
        return list(np.matmul(A,X)+self.Xm+np.matmul(B,u)+self.Um)

    def model(self,X,t,u):
        Tc = X[0]
        Ca = X[1]
        T = X[2]
        TO = X[3]
        return list(np.array(self.sisys(Tc,Ca,T,TO,u)).reshape(-1))

    def sym(self,tf,t0=0,dT=0.1):
        t=np.linspace(t0,tf,int((tf-t0)/(dT)))
        self.z_data = odeint(self.model,self.z0,t,args=(1.0,))
    
    #def sym_u(self,tf,t0=0,dT=0.1):
    def sym_step(self,tf,up=[[0,1]],t0=0,dT=0.1):
        n=int((tf-t0)/(dT))
        t=np.linspace(t0,tf,n)
        u=np.zeros(t.shape)
        up=np.array(up)
        for us in up:
            u[t>=us[0]]=us[1]

        Xs=np.empty((t.shape[0],self.sys.shape[0]))
        Xs[0,:]=self.z0

        temp_z0=self.z0

        for i in range (1,n):
            tspan=[t[i-1],t[i]]
            X = odeint(self.model,temp_z0,tspan,args=(u[i],))
            Xs[i,:]=X[0]
            temp_z0 = X[1]
        self.t_data=t
        self.z_data=Xs
        self.u_data=u

    def sym_u(self,controller,reference,tf=1000,t0=0,dT=0.1):
    #def sym_u(self,controller,tf=1000,t0=0,dT=0.1):
        n=int((tf-t0)/(dT))
        t=np.linspace(t0,tf,n)
        u=np.zeros(t.shape)

        Xs=np.empty((t.shape[0],self.sys.shape[0]))
        Xs[0,:]=self.z0

        temp_z0=self.z0

        for i in range (1,n):
            tspan=[t[i-1],t[i]]
            #Control input
            #function (model,t,Xs)
            ui=controller.u_gain_based(reference-temp_z0)


            X = odeint(self.model,temp_z0,tspan,args=(ui[0],))
            Xs[i,:]=X[0]
            u[i]=ui[0]
            temp_z0 = X[1]
        self.t_data=t
        self.z_data=Xs
        self.u_data=u


class Sys_():
    def __init__(self,eq_sys,state_variables,actuation_Variables,x0):
        
        #State variables
        self.Xsv=state_variables
        self.U=actuation_Variables

        self.sys=eq_sys

        self.sisys=lambdify(self.Xsv.tolist()+self.U.tolist(),self.sys)

        self.initial_conditions(x0)
        self.set_states_mean(Xm=self.z0,Um=0)

    def initial_conditions(self,z0=[50.5,2.113,88,0]):
        self.z0=z0

    def sym_lin_sys(self):
            self.slA=(sympy.Matrix(self.sys).jacobian(self.Xsv))
            self.slB=(sympy.Matrix(self.sys).jacobian(self.U))

    def lin_sys(self,Xc,Uc):
        Vtr=list(np.vstack((self.Xsv,self.U)).T[0])
        lA=self.slA.subs(dict(zip(Vtr,list(np.hstack((Xc,Uc))))))
        lB=self.slB.subs(dict(zip(Vtr,np.hstack((Xc,Uc)))))
        return lA,lB
    
    def set_states_mean(self,Xm,Um):
        self.Xm=Xm
        self.Um=Um
    
    def lmodel(self,X,t,u):
        A,B=self.lin_sys(X,u)
        return list(np.matmul(A,X)+self.Xm+np.matmul(B,u)+self.Um)

    def model(self,X,t,u):
        return list(np.array(self.sisys(*(X.tolist()+[u]))).reshape(-1))

    def sym(self,tf,t0=0,dT=0.1):
        t=np.linspace(t0,tf,int((tf-t0)/(dT)))
        self.z_data = odeint(self.model,self.z0,t,args=(1.0,))
    
    #def sym_u(self,tf,t0=0,dT=0.1):
    def sym_step(self,tf,up=[[0,1]],t0=0,dT=0.1):
        n=int((tf-t0)/(dT))
        t=np.linspace(t0,tf,n)
        u=np.zeros(t.shape)
        up=np.array(up)
        for us in up:
            u[t>=us[0]]=us[1]

        Xs=np.empty((t.shape[0],self.sys.shape[0]))
        Xs[0,:]=self.z0

        temp_z0=self.z0

        for i in range (1,n):
            tspan=[t[i-1],t[i]]
            X = odeint(self.model,temp_z0,tspan,args=(u[i],))
            Xs[i,:]=X[0]
            temp_z0 = X[1]
        self.t_data=t
        self.z_data=Xs
        self.u_data=u

    def sym_u(self,controller,reference,tf=1000,t0=0,dT=0.1):
    #def sym_u(self,controller,tf=1000,t0=0,dT=0.1):
        n=int((tf-t0)/(dT))
        t=np.linspace(t0,tf,n)
        u=np.zeros(t.shape)

        Xs=np.empty((t.shape[0],self.sys.shape[0]))
        Xs[0,:]=self.z0

        temp_z0=self.z0

        for i in range (1,n):
            tspan=[t[i-1],t[i]]
            #Control input
            #function (model,t,Xs)
            ui=controller.u_gain_based(reference-temp_z0)


            X = odeint(self.model,temp_z0,tspan,args=(ui[0],))
            Xs[i,:]=X[0]
            u[i]=ui[0]
            temp_z0 = X[1]
        self.t_data=t
        self.z_data=Xs
        self.u_data=u