# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: INTELIGENCIA ARTIFICIAL
Tema: MACHINE LEARNING
Alumno: Kevin Alain Rodriguez Cruz 
Profesor: Dr. Asdrúbal López Chau
Descripción: Graficar datos de un conjunto etiquetado
Created on Sun Oct 10 01:36:42 2021

@author: kevin
"""

import pandas as pd 
from matplotlib import pyplot as plt

#leer datos
datos = pd.read_csv('carros85.csv')
#seleccionamos las dos primeras dimensiones columnas pues
x = datos.iloc[:,1:3]
#seleccionamos las clases
y = datos.iloc[:,-1]

class GraficaClases:
    def plot2D(self,X,Y):
        """
        Grafica las clases en un conjunto de datos etiquetado

        Parameters
        ----------
        X : Dataframe
            Contiene atributos
        Y : Series pandas
            Contiene las etiquetas o clases

        Returns
        -------
        None.

        """
        colores = ['or','sg','hb','*y','pm']
        clases = list(set(Y))
        fig, axs = plt.subplots(1)
        clasesName = []
        for i in range (len(clases)):
            x = X.loc[Y==clases[i]].iloc[:,0]
            y = X.loc[Y==clases[i]].iloc[:,1]
            axs.plot(x,y,colores[i])
            clasesName.append(clases[i])
        nombreAtts = (X.columns)
        axs.set_xlabel(nombreAtts[0])
        axs.set_ylabel(nombreAtts[1])
        axs.legend(clasesName)
gc = GraficaClases() 
gc.plot2D(x,y)   