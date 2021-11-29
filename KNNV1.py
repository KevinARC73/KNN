# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: INTELIGENCIA ARTIFICIAL
Tema: Machine learning 
Alumno: Kevin Alain Rodriguez Cruz 
Profesor: Dr. Asdrúbal López Chau
Descripción: Clasificacion con K Vecinos mas cercanos 
K-Nearest Neighbourgs
Created on Sun Oct 10 16:48:28 2021

@author: kevin
"""
import numpy as np
import pandas as pd
class KNNV1:
    
        
    
    def fit(self, X,Y,K=6, distance='Euclidean'):
        """
        

        Parameters
        ----------
        X : dataframe
            Valores de atributos.
        Y : Series
            Etiquetas.
        K : int, optional
            Numero de vecinos mas cercanos, por default 3.
        distance : str, optional
            Distancia a utilizar. The default is 'Euclidean'.

        Returns
        -------
        None.

        """
        self.K=K
        self.distance = distance
        self.X = X
        self.Y= Y
        
    def distances(self,xi):
        """
        

        Parameters
        ----------
        xi : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        Xdatos = self.X.values
        distancias = []
        for j in range(len(Xdatos)):
            xj = Xdatos[j]
            dxi_xj = np.sqrt(np.sum(np.power(xi-xj,2)))
            distancias.append(dxi_xj)
        return distancias
    
    def getKminimunDistances(self, distancias):
        idxs = []
        for k in range (self.K):
            idx = np.argmin(distancias)
            idxs.append(idx)
            #distancias.remove(distancias[idx]) no eliminar ya ue cambian los indices 
            distancias[idx] = np.max(distancias)#correccion
        return idxs
    
    def claseMasFrecuente(self, idx):
        KClases = [self.Y[i] for i in idx]
        conjunto = list(set(KClases))
        frecuencia = [KClases.count(i) for i in conjunto]
        idx = np.argmax(frecuencia)
        return conjunto[idx] 
        
    def predict(self,X):
        if type(X) is pd.DataFrame:
            X = X.values
            ypred = []
            for xi in X:
                #calculo de tods las distancias a xi
                distancias = self.distances(xi)
                idx = self.getKminimunDistances(distancias)
                self.claseMasFrecuente(idx)
                ypred.append(self.claseMasFrecuente(idx))
        elif type(X) is np.ndarray:
            distancias = self.distances(X)
            idx = self.getKminimunDistances(distancias)
            ypred = self.claseMasFrecuente(idx)
        return ypred
            
        
