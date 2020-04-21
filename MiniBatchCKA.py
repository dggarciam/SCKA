#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 17:53:56 2020

@author: Daniel Guillermo Garcia, UNAL Mzls, Andrés Marino Álvarez Meza, UNAL-Mzls

Metric learning from Minibatch centered kernel alignment with orthogonalization
Enhanced version of the approach presented in:

Alvarez-Meza, A. M., Orozco-Gutierrez, A., & Castellanos-Dominguez, G. (2017). Kernel-based relevance analysis with enhanced interpretability for detection of brain activity patterns. Frontiers in neuroscience, 11, 550.

https://www.frontiersin.org/articles/10.3389/fnins.2017.00550/full

"""


#%%
#libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
import time
import scipy
#%%Optimizers----------------------------------------------------------------------------------------------------
##### checking#################################333
class RAdam(BaseEstimator, TransformerMixin):
    def __init__(self,learning_rate=0.01,min_lr=0.00001, 
                 beta_1= 0.9,beta_2=0.999,epsilon=1e-7):
        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._min_lr = min_lr
        self.iteration = 0
        self.m = None
        self.v = None
    
    def step(self,gradient,theta):
        if self.iteration == 0:
            t = self.iteration + 1
            self.iteration = t
            m = np.zeros(theta.shape)
            self.m = m
            v = np.zeros(theta.shape)
            self.v = v
        else:
            t = self.iteration
        beta1_t = np.power(self._beta1,t)
        beta2_t = np.power(self._beta2,t)
        sma_inf = 2.0/(1.0 - self._beta2) - 1.0
        sma_t = sma_inf - 2.0*t*beta2_t/(1.0 - beta2_t)
        self.m  = (self._beta1*self.m) + (1.0 - self._beta1)*gradient
        self.v  = (self._beta2*self.v) + (1.0 - self._beta2)*np.power(gradient,2)
        m_corr_t = self.m/(1.0 - beta1_t)
        if sma_t > 4:
            v_corr_t = np.sqrt(self.v/(1.0 - beta2_t))
            r_t = np.sqrt((sma_t - 4.0)/(sma_inf - 4.0) *
                          (sma_t - 2.0)/(sma_inf - 2.0) *
                           sma_inf/sma_t)
            self.step_ = self._lr*r_t*m_corr_t/(v_corr_t + self._epsilon)
            theta_t = theta-self.step_
        else:
            self.step_ = self._lr*m_corr_t
            theta_t = theta-self.step_
            
        self.iteration = self.iteration+1    
        return theta_t
#%% 
class Adam(BaseEstimator, TransformerMixin):
    def __init__(self,learning_rate=0.01,min_lr=0.00001, 
                 beta_1=0.9,beta_2=0.999,epsilon=1e-7):

        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._min_lr = min_lr
        self.iteration = 0
        self.m = None
        self.v = None
    
    
    def step(self,gradient,theta):
        if self.iteration == 0:
            self.iteration = self.iteration+1
            m = np.zeros(theta.shape)
            self.m = m
            v = np.zeros(theta.shape)
            self.v = v
        
        m = self._beta1*self.m + (1-self._beta1)*gradient
        v = self._beta2*self.v + (1-self._beta2)*np.power(gradient,2)
        m_hat = m/(1 - np.power(self._beta1,self.iteration))
        v_hat = v/(1 - np.power(self._beta2,self.iteration))
        self.step_ = self._lr*m_hat/(np.sqrt(v_hat)+self._epsilon)
        theta = theta - self.step_
        self.m = m
        self.v = v
        self.iteration = self.iteration+1
        return theta
    
#%% 
class NAdam(BaseEstimator, TransformerMixin):
    def __init__(self,learning_rate=0.01,min_lr=0.00001, 
                 beta_1=0.9,beta_2=0.999,epsilon=1e-7):

        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._min_lr = min_lr
        self.iteration = 0
        self.m = None
        self.v = None
    def step(self,gradient,theta):
        if self.iteration == 0:
            self.iteration = self.iteration+1
            m = np.zeros(theta.shape)
            self.m = m
            v = np.zeros(theta.shape)
            self.v = v
        m = self._beta1*self.m + (1-self._beta1)*gradient
        v = self._beta2*self.v + (1-self._beta2)*np.power(gradient,2)
        m_hat = m/(1 - np.power(self._beta1,self.iteration))+(1-self._beta1)*gradient/(1-np.power(self._beta1,self.iteration))
        v_hat = v/(1 - np.power(self._beta2,self.iteration))
        self.step_ = self._lr*m_hat/(np.sqrt(v_hat)+self._epsilon)
        theta = theta - self.step_
        self.m = m
        self.v = v
        self.iteration = self.iteration+1
        return theta
    
    
class Gd(BaseEstimator, TransformerMixin):
    def __init__(self,learning_rate=0.01,decay=0.1):
        self._lr = learning_rate
        self._decay = decay
        self.iteration = 0
        self.error_pocket = 0
        
    def step(self,gradient,theta):
        self._lr *= (1.0/(1.0+self._decay*self.iteration))
        self.step_ = self._lr*gradient
        theta_t = theta-self.step_
        self.iteration = self.iteration+1    
        return theta_t
    
            
#%%MiniBatch_CKA----------------------------------------------------------------------------------------------------
from scipy.spatial.distance import squareform
class MiniBatchCKA(BaseEstimator, TransformerMixin):
        
    def __init__(self, showCommandLine = False, optimizer = 'Adam',
                 learning_rate=0.01,decay=0.1,min_lr=0.00001,beta_1=0.9,beta_2=0.999,epsilon=1e-7,gamma=1,
                 min_grad = 1e-5, epoch = 50, batch = 30,Q = 0.9, init = 'random', max_errpocket = 30):
        #inicializar parametros requeridos
        self.showCommandLine = showCommandLine
        self.optimizer = optimizer #tipo optimizador en gradiente descendiente
        self.learning_rate = learning_rate #tasa de aprendizaje
        self.decay = decay #decaimiento tasa
        self.min_lr = min_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.min_grad = min_grad
        self.gamma = gamma
        
        self.epoch = epoch #numero de epocas gradiente descendiente
        self.batch = batch #tamaño del lote
        self.Q = Q #numero de dimensiones
        self.init = init #tipo inicializacion
        self.max_errpocket = max_errpocket #crieterio paciencia en cambio del gradiente descendiente
        
    def _orth2(self,x,a):
        #ortogonalizar matriz
        a = scipy.linalg.orth(a) #norma de A = Q
        #normalizar segun distancia entre puntos
        y = x.dot(a) #proyectar datos para calcular kernel
        d  = pairwise_distances(y) #calculo distancias
        d = (d+d.T)/2
        gammai = (np.sqrt(self.gamma)/np.median(squareform(d)))/a.shape[1]
        # 1/sigma^2 = gamma*1/dmediana**2
        an = gammai*a # exp(-0.5*gamma*d**2), norm(A)= gammai
        #max cka(k(x),l) s.t. norm(a,1 o 2) <= xi 
        return an 
              
    def _ckagrad(self,vecas,x,n,l,h): #funcion calculo gradiente
        a  = np.real(np.reshape(vecas,[np.shape(x)[1],-1],order='F')) #vectorizar matriz para aplicar gradiente
        y = x.dot(a) #proyectar datos para calcular kernel
        d  = pairwise_distances(y) #calculo distancias
        d = (d+d.T)/2
        k  = np.e**(-.5*d**2) #calculo del kernel
        ####gradiente##########################################
        ######################################################
        trkl = np.trace(k.dot(h).dot(l).dot(h))
        trkk = np.trace(k.dot(h).dot(k).dot(h))
        grad_lk = h.dot(l).dot(h)/trkl
        grad_k  = h.dot(k).dot(h)/trkk
        grad    = grad_lk - grad_k # sobra el 0.5
        p       = grad*k
        p = (p+p.T)/2
        grada = (x.T).dot( p-np.diagflat(p.dot(np.ones([n,1]))) ).dot(x.dot(a))
        grada = -4*np.real(np.ravel(grada,order='F')) #incluyendo negativo de la drivada de la distancia
        ##################################################################3
        #####funcion de costo############################################3
        f     = np.real(np.log(trkl)-np.log(trkk)/2.0)
        return f, grada, k, l

#%% Train---------------------------------------------------------------    
    def _ckatrain(self,X,L,labels): #X in N x P feature matrix, P features, N samples, L N x N target kernel,labels Nx1 vector
        startt = time.time()
        Q = self.Q #number of dimensions for cka-based projection
        max_errpocket = self.max_errpocket # max pocket error
        if self.init == 'pca': #projection matrix initialization
            red = PCA(n_components=Q)
            red.fit(X)
            A_i = red.components_.T
            del red
        elif self.init == 'random':
            if  Q < 1:
                red = PCA(n_components=Q)
                red.fit(X)
                A_i = np.random.rand(X.shape[1],red.components_.shape[0])
            else:    
                A_i = np.random.rand(X.shape[1],Q)
        else:
            A_i = self.init #init must be a P x Q matrix in this else
            
        epoch = self.epoch # number of epochs for mini batch gradient descent-based optimization
        batch = self.batch # batch size
        A    = A_i
        H = np.eye(batch) - (1.0/batch)*np.ones([batch,1])*np.ones([1,batch]) #matrix for centered kernel
        Fcost     = np.zeros((epoch*int(X.shape[0]/self.batch))) #cost along epoch
        normA = np.zeros((epoch*int(X.shape[0]/self.batch))) #cost along epoch
        fbest     = -np.inf
        
        #%% Optimization tipo de algoritmo gradiente descendiente
        if self.optimizer == 'Gd':
            self.opt_model = Gd(learning_rate=self.learning_rate,decay=self.decay)
        elif self.optimizer == 'Adam':
            self.opt_model = Adam(learning_rate=self.learning_rate,min_lr=self.min_lr, 
                             beta_1=self.beta_1,beta_2=self.beta_2,epsilon=self.epsilon)
        elif self.optimizer == 'RAdam':
            self.opt_model = RAdam(learning_rate=self.learning_rate,min_lr=self.min_lr, 
                             beta_1=self.beta_1,beta_2=self.beta_2,epsilon=self.epsilon)
        elif self.optimizer == 'NAdam':   
            self.opt_model = NAdam(learning_rate=self.learning_rate,min_lr=self.min_lr, 
                             beta_1=self.beta_1,beta_2=self.beta_2,epsilon=self.epsilon)
        #main loop for epochs
        jj = 0
        error_pocket = 0
        dderror_pocket = 0
        #pcaA=PCA(n_components=A.shape[1])
        for ii in np.arange(0,epoch):
            #print('Epoch %d of %d\n' % (ii,epoch))
            sss = StratifiedShuffleSplit(n_splits=int(X.shape[0]/self.batch),
                                        train_size=self.batch,test_size=X.shape[0]-self.batch)    
            for train_index, test_index in sss.split(X,labels):    
                #idxs = np.argsort(labels[train_index],kind='stable')
                idxs = np.argsort(labels[train_index])
                ##########################################
                #ortogonalizar y normalizar matriz inicial
                if jj == 0: 
                    A = self._orth2(X[train_index[idxs],:],A)
                    vecAs   = np.ravel(A,order='F') #vectorizing matrix for gradient descent opt
                fnew, gradf, k, l = self._ckagrad(vecAs,X[train_index[idxs],:],len(train_index),
                                                       L[train_index[idxs],:][:,train_index[idxs]],H)
                                                       # no es necesario retornar l
                                                       # no es necesario ingresar n no se usa
                
                Fcost[jj] = fnew
                #jj+=1# tiene que ir despues de comparar la funcion de costo
                #updating minibatch gradient descent
                if jj > 0: # solo hacer la comparación desde la iteracion 1->(2) 
                    #if Fcost[jj] < Fcost[jj-1]:
                    if fnew > fbest: #instanci actual tiene que ser mayor que la anterior
                        vbest = vecAs
                        grafbest = gradf
                        fbest = fnew 
                        error_pocket = 0
                    else:
                        error_pocket +=  1
                else:
                    fbest = fnew
                    vbest = vecAs
                    grafbest = gradf
                if error_pocket >= self.max_errpocket:
                    break 
                #scaling A
                ####paso del gradiente buscando minimizar funcion de costo##############
                vecAs = self.opt_model.step(grafbest,vbest)
                #####################################################################
                a  = np.real(np.reshape(vecAs,[np.shape(X)[1],-1],order='F'))
                #ortogonalizar matriz y normalizar matriz
                a = self._orth2(X[train_index[idxs],:],a)

                normA[jj] =  np.trace(a.T.dot(a))
                vecAs   = np.ravel(a,order='F')
                jj += 1
                if self.showCommandLine:
                    print('epoch:%d/%d--- fcostbest: %.7e --- fcost:%.7e --- errpoket:%d/%d --- elapsed:%.2f [s]\n' % (ii+1,epoch,fbest,fnew,error_pocket,max_errpocket,time.time()-startt))
        A  = np.real(np.reshape(vbest,[np.shape(X)[1],-1],order='F'))                        	
        return A, Fcost, normA
        
    #####declarar metodos fit y transform para acoplar clase con sklearn                               
    def fit(self,X,y, *_): 
        # X[samples x features]
        # y[labels x 1]
        y = np.squeeze(y)
        #idxs = np.argsort(y,kind='stable')
        #y = y[idxs]
        #X = X[idxs,:]
        KL = np.asmatrix(y).T == np.asmatrix(y)
        self.A, self.F, self.N = self._ckatrain(X,KL,y)
        return self 

    def transform(self, Xraw, *_):
        return  Xraw.dot(self.A)
              
        
    def fit_transform(self,X,y):
        self.fit(X,y)
        return  X.dot(self.A)
#%%
