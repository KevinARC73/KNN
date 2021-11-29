# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: INTELIGENCIA ARTIFICIAL
Tema:Machine Learning 
Alumno: Kevin Alain Rodriguez Cruz 
Profesor: Dr. Asdrúbal López Chau
Descripción: Prueba KNN
Created on Sun Oct 10 17:11:51 2021

@author: kevin
"""
import pandas as pd
from KNNV1 import KNNV1
import random 
import numpy as np
accs = []
it = int(input("Cuantas iteraciones?:"))

datos = pd.read_csv("carros85.csv")
X = datos.iloc[:,0:-1]#atributos
Y = datos.iloc[:,-1]#etiquetas
for i in range (it):    
    idx = list(range(X.shape[0]))
    random.shuffle(idx)#revolver indices
      
    #training set(seleccion pseudo-aleatoria)
    Xtr = pd.DataFrame(columns=X.columns)
    Ytr = []
    for i in range(int(len(idx)*.66)):
        Xtr = Xtr.append(X.iloc[idx[i],:])
        Ytr.append(Y[idx[i]])
        #test set
    Xtest = pd.DataFrame(columns=X.columns)
    Ytest = []
    for i in range(int(len(idx)*.66), len(idx)):
        Xtest = Xtest.append(X.iloc[idx[i],:])
        Ytest.append(Y[idx[i]])
        
    clf = KNNV1()
    clf.fit(Xtr, Ytr, K=6,distance="Euclidean")
    ypred = clf.predict(Xtest)
    #Medir que tan bien predice las etiquetas
    suma = np.sum(np.array(Ytest) == np.array(ypred))
    accuracy = suma/len(Ytest)
    print("Accuracy = {:.2f}".format(accuracy))
    ##accs.append(accuracy)
    #print("Acc: "+str(np.mean(accs)))
    #print("STD: "+ str(np.std(accs)))
    #Matriz de confusion
from MatrizConfusion import MatrizConfusion
mc = MatrizConfusion(Ytest, ypred)