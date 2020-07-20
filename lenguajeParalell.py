
########################################################
####                       API                      ####
########################################################

#############################################
#####       Paquetes necesarios         #####
#############################################
#Paquete necesario para crear graficas y usar funciones matematicas.
import networkx as nx
from community import community_louvain
#Ademas poner pip install python-louvain
#distancia de funciones de distribucion
import scipy
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
#Paquete necesario para  hacer las regresiones
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#import pymnet
#Paquete necesario para crear graficas y usar funciones matematicas.
import math
import numpy as np
import matplotlib.pyplot as plt
#Paquete excel
import pandas as pd
#Paquete para guardar las tuplas
import pickle
#Paquete conteo tiempo
from datetime import datetime
#from progress.bar import ChargingBar
from time import time
import ray
import time

from multiprocessing import Pool
import itertools



from  listasLenguaje import *


#-------------------------------------------------------------------------------
#            Funciones para  enlistar palabras (Textos_x_idioma)
#-------------------------------------------------------------------------------

def tiempo_desglosado(T):
    print(T.days, ' Dias ',T.seconds//3600, ' Horas ',(T.seconds % 3600)//60, ' Minutos ',T.seconds % 60, 'Segundos ',T.microseconds, ' Microsegundos.')

def Sample_N_palabras(P,N,Seed_Sample):
    t_0 = datetime.now()
    print('Fecha de inicio de seleccion: ',datetime.now())
    if N>len(P):
        print('N>len(P)')
    else:
        Muestra=np.random.RandomState(seed=Seed_Sample).choice(P,N,False)
    return Muestra
    print('Fecha de termino de seleccion: ',datetime.now())
    print('Seleccion terminada en: ',tiempo_desglosado(datetime.now()-t_0))

def Subconjunto_palabras(texto,N,Seed_Sample,Nuevo_nombre_texto):
    T_0=datetime.now()
    print('Fecha de inicio de la construccion de la muestra de '+str(N)+' palabras del texto '+texto+' con semilla '+str(Seed_Sample)+': ',T_0)
    P=pickle.load(open(texto+ ".p", "rb"  ))
    Q=Sample_N_palabras(P,N,Seed_Sample)
    P_Sub=' '.join(Q)
    myfile = open(Nuevo_nombre_texto+'Sample'+str(N)+'Seed'+str(Seed_Sample)+'.txt', 'w', encoding='utf-8')
    myfile.write(P_Sub)
    myfile.close()
    pickle.dump(Q, open( Nuevo_nombre_texto+'Sample'+str(N)+'Seed'+str(Seed_Sample)+".p", "wb" ), protocol=4)
    print('Subconjunto  de '+str(N)+' palabras del texto '+texto+' guardado en: ')
    print(Nuevo_nombre_texto+'Sample'+str(N)+'Seed'+str(Seed_Sample)+ ".txt"+' y ')
    print(Nuevo_nombre_texto+'Sample'+str(N)+'Seed'+str(Seed_Sample)+".p")
    print(len(Q))
    print('Fecha de conclusion de la construccion de la muestra de '+str(N)+' palabras del texto '+texto+' con semilla '+str(Seed_Sample)+': ',datetime.now())
    print('Construccion terminada en:')
    tiempo_desglosado(datetime.now()-T_0)

#Quita minusculas y ARTICULOS
def palabras_normalizadas(Nombre_corpus,idioma,limitador,Minusculas,Eliminar_Arti_y_prep):
    T=pickle.load(open(Nombre_corpus+ ".p", "rb"  ))
    T = ' '.join(T)
    if  Minusculas==True:
        T = T.lower()
    Palabras=T.split()
    R=[]
    for j in Palabras:
        if len(j)>limitador:
            if Eliminar_Arti_y_prep==True:
                if j not in ARTyPREP[idioma]:
                    R+=[j]
            else:
                R+=[j]
    R=list(sorted(set(R)))
    pickle.dump(R, open( Nombre_corpus+'N'+".p", "wb" ), protocol=4)
    print('Palabras bien guardado como: '+Nombre_corpus+'N'+".p")
    construir_corpus(Nombre_corpus+'N')
    print(str(len(R)), 'Palabras')
    return len(R)

#cambia formato a txt.
def construir_corpus(Nombre_corpus):
    T=pickle.load(open(Nombre_corpus+ ".p", "rb"  ))
    Y=' '.join(T)
    myfile = open(Nombre_corpus+'.txt', 'w', encoding='utf-8')
    myfile.write(Y)
    myfile.close()
    print('Corpus guardado como: '+Nombre_corpus+'.txt')

#Limpiamos el texto de signos de puntuacion y ordenamos las palabras en un vector
def palabras(texto,Delims):
    T=texto
    for delimitador in Delims:
      T = ' '.join(T.split(delimitador))
    Palabras=T.split()
    return Palabras

def Palabras_diferentes(T_extos,Nombre_corpus):
    r=set()
    for i in T_extos:
        Q=open(i+".txt", mode = 'r', encoding='utf-8').read()
        r=r.union(set(palabras(Q,DELIMITADORES['Todos'])))
    r=sorted(r)
    pickle.dump(list(r), open( Nombre_corpus+ ".p", "wb" ), protocol=4)
    print('Palabras diferentes guardado como: '+ Nombre_corpus+ ".p")
    construir_corpus(Nombre_corpus)
    print(str(len(r)), 'Palabras')
    return len(r)

#################################################
##########        Texto

#Primero debes buscar los caracteres faltantes si es que hay mas idiomas
#con la funcion Caracteres_faltantes_idioma(texto,Alfab,Delim)

#idioma in IDIOMAS                     , recordar no hacer para español
#limitador in [1,...,palabra_mas_larga], se sugiere limitador            =  1
#Minusculas in [True,False]            , se sugiere Minusculas           =  True
#Eliminar_Arti_y_prep in [True,False]  , se sugiere Eliminar_Arti_y_prep =  True
#N in [1,...,Cant_total_palabras]      , se sugiere N                    =  10000
#Seed_Sample in [1,...]                , se sugiere Seed_Sample          =  5     , para  que concuerde con el de español

def Textos_x_idioma(idioma,limitador,Minusculas,Eliminar_Arti_y_prep,N,Seed_Sample):
    T_0 = datetime.now()
    print('Iniciando a trabajar los textos del idioma  '+idioma+',limitador='+str(limitador)+',Minusculas='+str(Minusculas)+',Eliminar_Arti_y_prep='+str(Eliminar_Arti_y_prep)+',Seed_Sample='+str(Seed_Sample))
    T_extos=TEXTOS[idioma]
    Nombre_corpus=str(len(T_extos))+'Txt'+idioma
    Palabras_diferentes(T_extos,Nombre_corpus)
    Cant_p=palabras_normalizadas(Nombre_corpus,idioma,limitador,Minusculas,Eliminar_Arti_y_prep)
    texto=Nombre_corpus+'N'
    Subconjunto_palabras(texto,N,Seed_Sample,texto)
    print('Terminado el proceso de trabajar los textos del idioma  '+idioma+',limitador='+str(limitador)+',Minusculas='+str(Minusculas)+',Eliminar_Arti_y_prep='+str(Eliminar_Arti_y_prep)+',Seed_Sample='+str(Seed_Sample)+'  en:')
    tiempo_desglosado(datetime.now()-T_0)
    print('Fecha:')
    print(datetime.now())
    print('Palabras distintas del idioma '+idioma+':'+str(Cant_p))
    return Cant_p

#-------------------------------------------------------------------------------
#            Funciones para  generar redes (Redes_Partes_x_idioma)
#-------------------------------------------------------------------------------


########################################################
#####    Declarar funciones distancia palabras     #####
########################################################

#s1 y s2 son la palabra 1 y 2 respectivamente

#Distancia de Hamming
def h_dist(s1, s2):
    """Devuelve la distancia de Hamming entre dos secuencias de igual longitud"""
    if len(s1) != len(s2):
        raise ValueError("Indefinido para secuencias de distinta longitud")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

#Distancia de Levenshtein
def leven_dist(s1, s2):
    d=dict()
    for i in range(len(s1)+1):
       d[i]=dict()
       d[i][0]=i
    for i in range(len(s2)+1):
       d[0][i] = i
    for i in range(1, len(s1)+1):
       for j in range(1, len(s2)+1):
          d[i][j] = min(d[i][j-1]+1, d[i-1][j]+1, d[i-1][j-1]+(not s1[i-1] == s2[j-1]))
    return d[len(s1)][len(s2)]

#Distancia de Damerau-Levenshtein
def damerau_dist(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

def creaDataFrame(tamData):
    newData = np.zeros([tamData,tamData])
    return newData

@ray.remote
def distancias_x_palabra(x,P):
    vetor_distancia = np.array([damerau_dist(P[x], P[j]) for j in np.arange(len(P))])
    v =(x,vetor_distancia)
    return v

def get_matriz_distancias_parallelo(m1, m2, P, Corte, Nnodes):
    datosSalida = creaDataFrame(m2-m1)
    distancias = []
    ray.init()
    for i in np.arange(m1,m2):
        distancias.append(distancias_x_palabra.remote(i,P))
        if i%Corte==0:
            print(i,datetime.now())
        if (i%Nnodes==0):
            val_distancias = ray.get(distancias)
            for idata in range(len(val_distancias)):
                datosSalida[val_distancias[idata][0]] = val_distancias[idata][1]
            distancias = []
    val_distancias = ray.get(distancias)
    for idata in range(len(val_distancias)):
        datosSalida[val_distancias[idata][0]] = val_distancias[idata][1]
    ray.shutdown()
    return datosSalida

############################################################################
#############         Construccion de las redes            #################
############################################################################

def Matriz_Damerau_Partes(texto,RED,Parte_empiezo,Parte_final,Corte,idioma,Nnodes):
    T_0 = datetime.now()
    X=['Capa Gramatica','Capa Fonetica']
    Z=['Gram','Fon']
    P=pickle.load(open(texto+ ".p", "rb"  ))
    n=len(P)
    m1=Parte_empiezo
    m2=Parte_final
    Texto=texto+Z[X.index(RED)]
    if RED=='Capa Gramatica':
        print('[Gram][INICIO] Iniciando la creacion de la Matriz de Damerau la red '+RED+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final))
        T_r=datetime.now()
        print('__Fecha:',T_r)
        Datos= pd.DataFrame()
        print('>Cargando datos')
        print('__Fecha:',datetime.now())
        Datos= get_matriz_distancias_parallelo(m1, m2, P, Corte, Nnodes)
        #for i in np.arange(m1,m2):
        #    Datos[i]=np.array([damerau_dist(P[i], P[j]) for j in np.arange(n) ] )
        #    if i%Corte==0:
        #        print(i,datetime.now())
        pickle.dump(Datos, open(Texto+'MatrizDam'+str(m1)+'_'+str(m2)+".p", "wb" ), protocol=4)
        print('-Matriz de adyacencia de la red '+Texto+' guardada en:')
        print(Texto+'MatrizDam'+str(m1)+'_'+str(m2)+".p")
        print('[Gram][FIN] Fecha de conclusion de construccion de la Matriz de Damerau '+RED+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final),datetime.now())
        print('->Terminado en:')
        tiempo_desglosado(datetime.now()-T_r)
    elif RED=='Capa Fonetica':
        print('[Fon][INICIO] Iniciando la creacion de la Matriz de Damerau la red '+RED+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final))
        T_r=datetime.now()
        print('__Fecha:',T_r)
        PTrad=[Traduccion[idioma].transliterate(l) for l in P]
        Datos= pd.DataFrame()
        print('>Cargando datos')
        print('__Fecha:',datetime.now())
        Datos= get_matriz_distancias_parallelo(m1, m2, P, Corte, Nnodes)
        # for i in np.arange(m1,m2):
        #     Datos[i]=np.array([damerau_dist(PTrad[i], PTrad[j]) for j in np.arange(n) ] )
        #     if i%Corte==0:
        #         print(i,datetime.now())
        pickle.dump(Datos, open(Texto+'MatrizDam'+str(m1)+'_'+str(m2)+".p", "wb" ), protocol=4)
        print('-Matriz de adyacencia de la red '+Texto+' guardada en:')
        print(Texto+'MatrizDam'+str(m1)+'_'+str(m2)+".p")
        print('[Fon][FIN] Fecha de conclusion de construccion de la Matriz de Damerau '+RED+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final),datetime.now())
        print('->Terminado en:')
        tiempo_desglosado(datetime.now()-T_r)

def Red_Partes_Desde_Matriz_Damerau(texto,RED,Rangos,Parte_empiezo,Parte_final,Corte,idioma):
    S1=','.join([str(elem) for elem in Rangos])
    X=['Capa Gramatica','Capa Fonetica']
    Z=['Gram','Fon']
    P=pickle.load(open(texto+ ".p", "rb"  ))
    n=len(P)
    m1=Parte_empiezo
    m2=Parte_final
    Texto=texto+Z[X.index(RED)]
    if RED=='Capa Gramatica':
        print('[Gram][INICIO] Creando la red de la '+RED+' con rangos '+S1+' del texto '+texto+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final))
        print('__Fecha:',datetime.now())
        T_Capa=datetime.now()
        dicName = {}
        for iname in range(len(P)):
            dicName[iname] = P[iname]
        for Rango in Rangos:
            print('>>Iniciando la creacion de la red '+RED+' con rango '+str(Rango)+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final))
            T_r=datetime.now()
            print('__Fecha:',T_r)
            print('>Cargando la Matriz de Damerau la red '+RED+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final))
            print('__Fecha:',datetime.now())
            MDdata=pickle.load(open(Texto+'MatrizDam'+str(m1)+'_'+str(m2)+".p", "rb"  ))
            print('>Cargando datos con distancia permitida de enlace '+str(Rango))
            print('__Fecha:',datetime.now())
            #MDdata = MDdata.to_numpy()
            MDdata[MDdata==0] = 1
            MDdata[MDdata>Rango] = 0
            np.fill_diagonal(MDdata, 0)
            G = nx.from_numpy_matrix(MDdata)
            G = nx.relabel_nodes(G, dicName)
            pickle.dump(G, open(Texto+'R'+str(Rango)+'Red'+str(m1)+'_'+str(m2)+".p", "wb" ), protocol=4)
            print('-Subred de la red '+RED+' con rango '+str(Rango)+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final)+' guardada en:')
            print(Texto+'R'+str(Rango)+'Red'+str(m1)+'_'+str(m2)+".p")
            print('Numero de nodos:'+str(nx.number_of_nodes(G)))
            print('Numero de enlaces:'+str(nx.number_of_edges(G)))
            print('__Fecha de conclusion de la construccion la red '+RED+' con rango '+str(Rango)+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final),datetime.now())
            print('->Terminado en:')
            tiempo_desglosado(datetime.now()-T_r)
        print('[Gram][FIN] Fecha de conclusion de la red '+RED+' con rangos '+S1+' del texto '+texto+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final),datetime.now())
        print('--->Terminado en:')
        tiempo_desglosado(datetime.now()-T_Capa)
    elif RED=='Capa Fonetica':
        print('[Fon][INICIO] Creando la red de la '+RED+' con rangos '+S1+' del texto '+texto+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final))
        T_Capa=datetime.now()
        print('__Fecha:',T_Capa)
        PTrad=[Traduccion[idioma].transliterate(l) for l in P]
        PT0=[(PTrad[h],P[h]) for h in range(n)]
        dicName = {}
        for iname in range(len(PT0)):
            dicName[iname] = PT0[iname]
        for Rango in Rangos:
            print('>>Iniciando la creacion de la red '+RED+' con rango '+str(Rango)+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final))
            T_r=datetime.now()
            print('__Fecha:',T_r)
            print('>Cargando la Matriz de Damerau la red '+RED+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final))
            print('__Fecha:',datetime.now())
            MDdata=pickle.load(open(Texto+'MatrizDam'+str(m1)+'_'+str(m2)+".p", "rb"  ))
            print('>Cargando datos con distancia permitida de enlace '+str(Rango))
            print('__Fecha:',datetime.now())
            #MDdata = MDdata.to_numpy()
            MDdata[MDdata==0] = 1
            MDdata[MDdata>Rango] = 0
            np.fill_diagonal(MDdata, 0)
            G = nx.from_numpy_matrix(MDdata)
            G = nx.relabel_nodes(G, dicName)
            pickle.dump(G, open(Texto+'R'+str(Rango)+'Red'+str(m1)+'_'+str(m2)+".p", "wb" ), protocol=4)
            print('-Subred de la red '+RED+' con rango '+str(Rango)+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final)+' guardada en:')
            print(Texto+'R'+str(Rango)+'Red'+str(m1)+'_'+str(m2)+".p")
            print('Numero de nodos:'+str(nx.number_of_nodes(G)))
            print('Numero de enlaces:'+str(nx.number_of_edges(G)))
            print('__Fecha de conclusion de la construccion la red '+RED+' con rango '+str(Rango)+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final),datetime.now())
            print('->Terminado en:')
            tiempo_desglosado(datetime.now()-T_r)
        print('[Fon][FIN] Fecha de conclusion de la red '+RED+' con rangos '+S1+' del texto '+texto+' con indices desde '+str(Parte_empiezo)+' hasta '+str(Parte_final),datetime.now())
        print('--->Terminado en:')
        tiempo_desglosado(datetime.now()-T_Capa)

#################################################
##########        Redes

#idioma in IDIOMAS                                       , recordar no hacer para español
#N in [1,...,Cant_total_palabras]                        , se sugiere N             = 10000
#Seed_Sample in [1,...,Cant_total_palabras]              , se sugiere Seed_Sample   = 5          , para  que concuerde con el de español
#Seed_Shuff in [1,...,Cant_total_palabra                         , se sugiere Seed_Shuff         = 0                                                 , porque no se  ha hecho un intercambio interno de caracteres
#Parte_empiezo= una_de_las_cotas_izquierdas_de_particion , se sugiere Parte_empiezo = 0          , el  valor minimo es 0
#Parte_final= una_de_las_cotas_derechas_de_particion     , se sugiere Parte_final   = N          , el valor maximo es Cant_total_palabras
#Corte in np.arange(100)+1                               , se sugiere Corte         = N//10      , 10% de los datos
#Rangos=[n1,n2,...]                                      , se sugiere Rangos        = [1,2,3,4,5], pero puede ser cualquier lista en el orden que sea
#V_Lim_by_R=[[0,n1(i),...,nk(i),N] for i in Rangos]      , se sugiere V_Lim_by_R    = [[0,N] for i in Rangos]


def Redes_Partes_x_idioma(idioma,N,Seed_Sample,Parte_empiezo,Parte_final,Corte,Rangos,Nnodes):
    T_0 = datetime.now()
    print('[INICIO][Red_partes]Iniciando a construir parte de la red del idioma '+idioma+' desde '+str(Parte_empiezo)+' hasta '+str(Parte_final)+',Corte='+str(Corte)+',Rangos='+str(Rangos))
    print('Fecha:',datetime.now())
    texto=str(len(TEXTOS[idioma]))+'Txt'+idioma+'N'+'Sample'+str(N)+'Seed'+str(Seed_Sample)
    for RED in ['Capa Gramatica','Capa Fonetica']:
        Matriz_Damerau_Partes(texto,RED,Parte_empiezo,Parte_final,Corte,idioma,Nnodes)
        Red_Partes_Desde_Matriz_Damerau(texto,RED,Rangos,Parte_empiezo,Parte_final,Corte,idioma)
    print('[Fin][Red_partes]Iniciando a construir parte de la red del idioma '+idioma+' desde '+str(Parte_empiezo)+' hasta '+str(Parte_final)+',Corte='+str(Corte)+',Rangos='+str(Rangos)+' en:')
    print('Fecha:',datetime.now())
    print('[Fin][Red_partes]Tiempo de construccion:',datetime.now())
    tiempo_desglosado(datetime.now()-T_0)


#-------------------------------------------------------------------------------
#            Funciones para  generar redes (Redes_x_idioma)
#-------------------------------------------------------------------------------


def Red_Desde_Partes(texto,RED,Rangos,V_Lim_by_R,idioma):
    S1=''
    for v in Rangos:
        S1=S1+str(v)
    X=['Capa Gramatica','Capa Fonetica']
    Z=['Gram','Fon']
    P=pickle.load(open(texto+ ".p", "rb"  ))
    n=len(P)
    Texto=texto+Z[X.index(RED)]
    if RED=='Capa Gramatica':
        print('[Gram][INICIO] Creando la red de la '+RED+' con rangos '+S1+' del texto '+texto)
        print('__Fecha:',datetime.now())
        T_Capa=datetime.now()
        for Rango in Rangos:
            E1=''
            for v in Rangos:
                E1=E1+str(v)
            print('>>Iniciando la creacion de la red '+RED+' con rango '+str(Rango)+' con limites de indices '+E1)
            T_r=datetime.now()
            G=nx.Graph()
            G.add_nodes_from(P)
            for l in np.arange(len(V_Lim_by_R[Rangos.index(Rango)])-1):
                print('>Cargando subred la red '+RED+' con indices desde '+str(V_Lim_by_R[Rangos.index(Rango)][l])+' hasta '+str(V_Lim_by_R[Rangos.index(Rango)][l+1]))
                print('__Fecha:',datetime.now())
                SubRed=pickle.load(open(Texto+'R'+str(Rango)+'Red'+str(V_Lim_by_R[Rangos.index(Rango)][l])+'_'+str(V_Lim_by_R[Rangos.index(Rango)][l+1])+".p", "rb"  ))
                G.add_edges_from(SubRed.edges())
                del(SubRed)
            pickle.dump(G, open(Texto+'R'+str(Rango)+'Red'+".p", "wb" ), protocol=4)
            print('-Red de la '+RED+' con rango '+str(Rango)+' guardada en:')
            print(Texto+'R'+str(Rango)+'Red'+".p")
            print('Numero de nodos:'+str(nx.number_of_nodes(G)))
            print('Numero de enlaces:'+str(nx.number_of_edges(G)))
            print('__Fecha de conclusion de la construccion la red '+RED+' con rango '+str(Rango),datetime.now())
            print('->Terminado en:')
            tiempo_desglosado(datetime.now()-T_r)
        print('[Gram][FIN] Fecha de conclusion de la red '+RED+' con rangos '+S1+' del texto '+texto,datetime.now())
        print('--->Terminado en:')
        tiempo_desglosado(datetime.now()-T_Capa)
    elif RED=='Capa Fonetica':
        print('[Fon][INICIO] Creando la red de la '+RED+' con rangos '+S1+' del texto '+texto)
        print('__Fecha:',datetime.now())
        T_Capa=datetime.now()
        PTrad=[Traduccion[idioma].transliterate(l) for l in P]
        PT0=[(PTrad[h],P[h]) for h in range(n)]
        for Rango in Rangos:
            E1=''
            for v in Rangos:
                E1=E1+str(v)
            print('>>Iniciando la creacion de la red '+RED+' con rango '+str(Rango)+' con limites de indices '+E1)
            T_r=datetime.now()
            G=nx.Graph()
            G.add_nodes_from(PT0)
            for l in np.arange(len(V_Lim_by_R[Rangos.index(Rango)])-1):
                print('>Cargando subred la red '+RED+' con indices desde '+str(V_Lim_by_R[Rangos.index(Rango)][l])+' hasta '+str(V_Lim_by_R[Rangos.index(Rango)][l+1]))
                print('__Fecha:',datetime.now())
                SubRed=pickle.load(open(Texto+'R'+str(Rango)+'Red'+str(V_Lim_by_R[Rangos.index(Rango)][l])+'_'+str(V_Lim_by_R[Rangos.index(Rango)][l+1])+".p", "rb"  ))
                G.add_edges_from(SubRed.edges())
                del(SubRed)
            pickle.dump(G, open(Texto+'R'+str(Rango)+'Red'+".p", "wb" ), protocol=4)
            print('-Red de la '+RED+' con rango '+str(Rango)+' guardada en:')
            print(Texto+'R'+str(Rango)+'Red'+".p")
            print('Numero de nodos:'+str(nx.number_of_nodes(G)))
            print('Numero de enlaces:'+str(nx.number_of_edges(G)))
            print('__Fecha de conclusion de la construccion la red '+RED+' con rango '+str(Rango),datetime.now())
            print('->Terminado en:')
            tiempo_desglosado(datetime.now()-T_r)
        print('[Fon][FIN] Fecha de conclusion de la red '+RED+' con rangos '+S1+' del texto '+texto,datetime.now())
        print('--->Terminado en:')
        tiempo_desglosado(datetime.now()-T_Capa)


def Redes_x_idioma(idioma,N,Seed_Sample,V_Lim_by_R,Rangos):
    T_0 = datetime.now()
    print('[INICIO][Redes] Iniciando a construir las redes del idioma '+idioma+',N='+str(N)+',Rangos='+str(Rangos)+'Seed_Sample='+str(Seed_Sample)+'V_Lim_by_R='+str(V_Lim_by_R)+' desde partes.')
    print('Fecha:',datetime.now())
    texto=str(len(TEXTOS[idioma]))+'Txt'+idioma+'N'+'Sample'+str(N)+'Seed'+str(Seed_Sample)
    for RED in ['Capa Gramatica','Capa Fonetica']:
        Red_Desde_Partes(texto,RED,Rangos,V_Lim_by_R,idioma)
    print('[Fin][Redes] Redes del idioma '+idioma+',N='+str(N)+',Rangos='+str(Rangos)+'Seed_Sample='+str(Seed_Sample)+' desde partes, construidas:')
    print('Fecha:',datetime.now())
    print('[Fin][Redes] Tiempo de construccion:',datetime.now())
    tiempo_desglosado(datetime.now()-T_0)


#-------------------------------------------------------------------------------
#            Funciones para  generar redes multiplex (Red_Multiplex_x_idioma)
#-------------------------------------------------------------------------------

def Red_Multiplex(texto,Seed_Sample,Seed_Shuff,Rangos,idioma):
    T_0 = datetime.now()
    if len(Rangos)==1:
         S1=str(Rangos[0])
         R=str(Rangos[0])
    else:
        S1=','.join([str(elem) for elem in Rangos])
        R='_'.join([str(elem) for elem in Rangos])
    if Seed_Sample==0:
        Texto=texto
    else:
        if Seed_Shuff==0:
            Texto=texto+'Seed'+str(Seed_Sample)
        else:
            Texto=texto+'Seed'+str(Seed_Sample)+'Shuf_Seed'+str(Seed_Shuff)
    print('[INICIO] Creando las redes multiplex de las redes con rangos '+S1+' del texto '+Texto)
    print('__Fecha:',datetime.now())
    for Rango in Rangos:
        t_r=datetime.now()
        G=nx.Graph()
        H=pickle.load(open(Texto+'Gram'+'R'+str(Rango)+ "Red.p", "rb"  ))
        P=list(H.nodes())
        PTrad=[]
        Numerico = np.char.isnumeric(P)
        for l in range(len(P)):
            if not Numerico[l]:
                PTrad.append(Traduccion[idioma].transliterate(P[l]))
        n=len(PTrad)
        PT0=[(PTrad[h],P[h]) for h in range(n)]
        G.add_nodes_from(H.nodes())
        G.add_edges_from(H.edges())
        H=pickle.load(open(Texto+'Fon'+'R'+str(Rango)+ "Red.p", "rb"  ))
        G.add_nodes_from(H.nodes())
        G.add_edges_from(H.edges())
        G.add_edges_from([(P[j],PT0[j]) for j in range(n)] )
        pickle.dump(G, open(Texto+'Multi'+'R'+str(Rango)+"Red.p", "wb" ), protocol=4)
        print('Red multiplex con rango '+str(Rango)+' del texto '+Texto+' guardada en:')
        print(Texto+'Multi'+'R'+str(Rango)+"Red.p")
        print('Numero de nodos:'+str(nx.number_of_nodes(G)))
        print('Numero de enlaces:'+str(nx.number_of_edges(G)))
        print('->Tiempo utilizado en el rango '+str(Rango)+':')
        tiempo_desglosado(datetime.now()-T_0)
    print('[FINAL]->Proceso completo de los rangos '+S1+' terminado en:')
    tiempo_desglosado(datetime.now()-T_0)

def Red_Multiplex_x_idioma(idioma,N,Seed_Sample,Seed_Shuff,Rangos):
    texto=str(len(TEXTOS[idioma]))+'Txt'+idioma+'N'+'Sample'+str(N)
    Red_Multiplex(texto,Seed_Sample,Seed_Shuff,Rangos,idioma)

#-------------------------------------------------------------------------------
#            Funciones para  generar medidas de red (Medidas_x_idioma)
#-------------------------------------------------------------------------------

from sklearn import linear_model
lm = linear_model.LinearRegression()

def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x

def _betmap(G_normalized_weight_sources_tuple):
    """Pool for multiprocess only accepts functions with one argument.
    This function uses a tuple as its only argument. We use a named tuple for
    python 3 compatibility, and then unpack it when we send it to
    `betweenness_centrality_source`
    """
    return nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)

def betweenness_centrality_parallel(G, processes=40):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.map(_betmap,
                  zip([G] * num_chunks,
                      [True] * num_chunks,
                      [None] * num_chunks,
                      node_chunks))

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c

def Etiquetas_aleatorio(texto,RED,Seed_Sample,Seed_Shuff,Cant_total_palabras,Cant_muestra_palabras,idioma):
    Z=['Multi','Gram','Fon']
    X=['Red Multiplex','Capa Gramatica','Capa Fonetica']
    caso=''
    if RED in X:
        caso=Z[X.index(RED)]
    if Seed_Sample==0:
        Texto0=texto+caso
        TituloT0='Texto '+texto
        Muestra0=' de '+str(Cant_total_palabras)+' palabras en '+idioma+'.'
    else:
        Muestra0=' de '+str(Cant_muestra_palabras)+' palabras de '+str(Cant_total_palabras)+' en '+idioma+'.'
        if Seed_Shuff==0:
            Texto0=texto+'Seed'+str(Seed_Sample)+caso
            TituloT0='Texto '+texto+' semilla sample '+str(Seed_Sample)
        else:
            Texto0=texto+'Seed'+str(Seed_Sample)+'Shuf_Seed'+str(Seed_Shuff)+caso
            TituloT0='Texto '+texto+' semilla sample '+str(Seed_Sample)+' semilla shuffle '+str(Seed_Shuff)
    return Texto0,TituloT0,Muestra0

def Densidad_Metrica(texto,RED,Rangos,metrica):
    if len(Rangos)==1:
        S1=str(Rangos[0])
    else:
        S1=','.join([str(elem) for elem in Rangos])
    X=['Red Multiplex','Capa Gramatica','Capa Fonetica']
    Z=['Multi','Gram','Fon']
    print('['+Z[X.index(RED)]+'][INICIO][Densidad  '+metrica+']'+RED+' con rangos '+S1+' del texto '+texto)
    print('[INICIO] Fecha:',datetime.now())
    T_Capa=datetime.now()
    Texto=texto+Z[X.index(RED)]
    if metrica=='Grados':
        for Rango in Rangos:
            print('>>Iniciando para el rango '+str(Rango))
            T_r=datetime.now()
            DF=pickle.load(open(Texto+'R'+str(Rango)+'Grados'+".p", "rb"  ))
            Densidad_0= DF.Grados.value_counts(True)
            pickle.dump(Densidad_0, open(Texto+'R'+str(Rango)+'DensGrados'+ ".p", "wb" ), protocol=4)
            print('>>] Terminado para el rango  '+str(Rango)+' en:')
            tiempo_desglosado(datetime.now()-T_r)
            print('----]Fecha:',datetime.now())
    elif metrica=='CentrBet':
        for Rango in Rangos:
            print('>>Iniciando para el rango '+str(Rango))
            T_r=datetime.now()
            DF=pickle.load(open(Texto+'R'+str(Rango)+'CentrBet'+".p", "rb"  ))
            Densidad_0= DF.CentralidadB.value_counts(True)
            pickle.dump(Densidad_0, open(Texto+'R'+str(Rango)+'DensCentrBet'+ ".p", "wb" ), protocol=4)
            print('>>] Terminado para el rango  '+str(Rango)+' en:')
            tiempo_desglosado(datetime.now()-T_r)
            print('----]Fecha:',datetime.now())
    else:
        print('Metrica no valida')
    print('-Densidad de '+metrica+' de la '+RED+' con rangos '+S1+' guardadas en:')
    print(Texto+'R'+S1+metrica+".p")
    print('['+Z[X.index(RED)]+'][FIN][Densidad  '+metrica+']'+RED+' con rangos '+S1+' del texto '+texto+' finalizada en:')
    tiempo_desglosado(datetime.now()-T_Capa)
    print('[FIN][Densidad  '+metrica+'] Fecha:',datetime.now())

@ray.remote
def get_grado(Texto, Rango):
    try:
        print('>>Iniciando para el rango '+str(Rango))
        T_r=datetime.now()
        G=pickle.load(open(Texto+'R'+str(Rango)+'Red'+".p", "rb"  ))
        Gr=G.degree()
        DF=pd.DataFrame.from_dict(dict(Gr),orient='index', columns=['Grados'])
        pickle.dump(DF, open(Texto+'R'+str(Rango)+'Grados'+".p", "wb" ), protocol=4)
        print('>>] Terminado para el rango  '+str(Rango)+' en:')
        tiempo_desglosado(datetime.now()-T_r)
        print('----]Fecha:',datetime.now())
        return ('Grado '+ str(Rango))
    except Exception as e:
        return ('Grado '+ str(Rango) + "-->" + str(e))

@ray.remote
def get_clustering(Texto, Rango):
    try:
        print('>>Iniciando para el rango '+str(Rango))
        T_r=datetime.now()
        G=pickle.load(open(Texto+'R'+str(Rango)+'Red'+".p", "rb"  ))
        Clustering=nx.clustering(G)
        DF=pd.DataFrame.from_dict(Clustering,orient='index', columns=['Clustering'])
        pickle.dump(DF, open(Texto+'R'+str(Rango)+'Clust'+".p", "wb" ), protocol=4)
        print('>>] Terminado para el rango  '+str(Rango)+' en:')
        tiempo_desglosado(datetime.now()-T_r)
        print('----]Fecha:',datetime.now())
        return ('clustering '+ str(Rango))
    except Exception as e:
        return ('clustering '+ str(Rango) + "-->" + str(e))

@ray.remote
def get_AND(Texto, Rango):
    try:
        print('>>Iniciando para el rango '+str(Rango))
        T_r=datetime.now()
        G=pickle.load(open(Texto+'R'+str(Rango)+'Red'+".p", "rb"  ))
        Average_ND=nx.average_neighbor_degree(G)
        DF=pd.DataFrame.from_dict(Average_ND,orient='index', columns=['AND'])
        pickle.dump(DF, open(Texto+'R'+str(Rango)+'AND'+".p", "wb" ), protocol=4)
        print('>>] Terminado para el rango  '+str(Rango)+' en:')
        tiempo_desglosado(datetime.now()-T_r)
        print('----]Fecha:',datetime.now())
        return ('AND '+ str(Rango))
    except Exception as e:
        return ('AND '+ str(Rango) + "-->" + str(e))

def Metrica(texto,RED,Rangos,metrica,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff,VdeV,Bineo,TipoBin,Porcentaje):

    if len(Rangos)==1:
        S1=str(Rangos[0])
    else:
        S1=','.join([str(elem) for elem in Rangos])
    X=['Red Multiplex','Capa Gramatica','Capa Fonetica']
    Z=['Multi','Gram','Fon']
    Texto=texto+Z[X.index(RED)]
    print('['+Z[X.index(RED)]+'][INICIO]['+metrica+']'+RED+' con rangos '+S1+' del texto '+texto)
    print('[INICIO] Fecha:',datetime.now())
    T_Capa=datetime.now()
    result_metrica =[]
    if metrica=='Grados':
        for Rango in Rangos:
            result_metrica.append(get_grado.remote(Texto, Rango))
    elif metrica=='Clust':
        for Rango in Rangos:
            result_metrica.append(get_clustering.remote(Texto, Rango))
    elif metrica=='AND':
        for Rango in Rangos:
            result_metrica.append(get_AND.remote(Texto, Rango))
    else:
        print('Metrica no valida')
    results = ray.get(result_metrica)
    print(results)
    if metrica in ['Grados']:
        Densidad_Metrica(texto,RED,Rangos,metrica)
        for Regresion  in ['Ml','Mv']:
            Densidad_Metrica_Regresiones(Regresion,texto,RED,Rangos,metrica,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff,VdeV,Bineo,TipoBin,Porcentaje)
    print('-'+metrica+' de la '+RED+' con rangos '+S1+' guardadas en:')
    print(Texto+'R'+S1+metrica+".p")
    print('['+Z[X.index(RED)]+'][FIN]['+metrica+']'+RED+' con rangos '+S1+' del texto '+texto+' finalizada en:')
    tiempo_desglosado(datetime.now()-T_Capa)
    print('[FIN]['+metrica+'] Fecha:',datetime.now())
    return metrica

def MetricaBet(texto,RED,Rangos,metrica,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff,VdeV,Bineo,TipoBin,Porcentaje):

    if len(Rangos)==1:
        S1=str(Rangos[0])
    else:
        S1=','.join([str(elem) for elem in Rangos])
    X=['Red Multiplex','Capa Gramatica','Capa Fonetica']
    Z=['Multi','Gram','Fon']
    Texto=texto+Z[X.index(RED)]
    print('['+Z[X.index(RED)]+'][INICIO]['+metrica+']'+RED+' con rangos '+S1+' del texto '+texto)
    print('[INICIO] Fecha:',datetime.now())
    T_Capa=datetime.now()

    if metrica=='CentrBet':
        for Rango in Rangos:
            print('>>Iniciando para el rango '+str(Rango))
            T_r=datetime.now()
            # G=pickle.load(open(Texto+'R'+str(Rango)+'Red'+".p", "rb"  ))
            # Centralidad= betweenness_centrality_parallel(G)
            # DF=pd.DataFrame.from_dict(Centralidad,orient='index', columns=['CentralidadB'])
            # pickle.dump(DF, open(Texto+'R'+str(Rango)+'CentrBet'+".p", "wb" ), protocol=4)
            print('>>] Terminado para el rango  '+str(Rango)+' en:')
            tiempo_desglosado(datetime.now()-T_r)
            print('----]Fecha:',datetime.now())
    if metrica in ['CentrBet']:
        Densidad_Metrica(texto,RED,Rangos,metrica)
        for Regresion  in ['Ml','Mv']:
            Densidad_Metrica_Regresiones(Regresion,texto,RED,Rangos,metrica,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff,VdeV,Bineo,TipoBin,Porcentaje)
    print('-'+metrica+' de la '+RED+' con rangos '+S1+' guardadas en:')
    print(Texto+'R'+S1+metrica+".p")
    print('['+Z[X.index(RED)]+'][FIN]['+metrica+']'+RED+' con rangos '+S1+' del texto '+texto+' finalizada en:')
    tiempo_desglosado(datetime.now()-T_Capa)
    print('[FIN]['+metrica+'] Fecha:',datetime.now())
    return metrica

def Split_gen(Vector,separador):
    if len(Vector)==1:
        S1=str(Vector[0])
    else:
        S1=separador.join([str(elem) for elem in Vector])
    return S1

def Formula_Coeficiente_Exp_MaximaV(Vector):
    minimo=min(Vector)
    n=len(Vector)
    A=1+n/sum(np.log(np.array(Vector)/minimo))
    return A

def Formula_Coeficiente_Factor_MaximaV(Vector,Vector2,Alfa_0):
    V_1=Vector
    #V_2=np.log(V_1)
    #Y_1=np.exp(Alfa_0*V_2)
    Y_1=V_1**Alfa_0
    W=np.array(Vector2)*Y_1
    MediaCoef=np.mean(W)
    return MediaCoef

def Aprox_linal_Newman(Vector,Vector2,Alfa1):
    V=np.array(Vector)
    #C_0=Formula_Coeficiente_Factor_MaximaV(Vector,Vector2,Alfa1)
    #V2=np.log(V)
    #Y=((Alfa1-1)/min(V))*np.exp(-Alfa1*V2)
    Y=((Alfa1-1)/min(V))*((V/min(V))**(-Alfa1))
    return Y

def Binear(DatosX,DatosY,TipoBin,Porcentaje):
    DatosY=[DatosY[x] for  x in range(len(DatosX)) if  DatosX[x]>0]
    DatosX=[x for x in DatosX if  x>0]
    N=len(DatosX)
    MIN,MAX=min(DatosX),max(DatosX)
    if int(N*(Porcentaje/100))<10:
        Nbin=N
    else:
        Nbin=int(N*(Porcentaje/100))
    Y=np.zeros(Nbin)
    Indices_malos=[]
    if TipoBin=='Log':
        bins = np.linspace(np.log10(MIN), np.log10(MAX), Nbin)
        X=10**bins
        for i in range(Nbin):
            if 0<i<Nbin-1:
                Conjunto=[DatosY[j] for j in range(N) if  10**(np.mean(bins[i-1:i+1]))<DatosX[j]<=10**(np.mean(bins[i:i+2]))]
                if len(Conjunto)==0:
                    Indices_malos+=[i]
                else:
                    Y[i]=np.mean(Conjunto)
            elif i==0:
                Y[i]=np.mean([DatosY[j] for j in range(N) if  DatosX[j]<=10**(np.mean(bins[i:i+2]))])
            else:
                Y[i]=np.mean([DatosY[j] for j in range(N) if  10**(np.mean(bins[i-1:i+1]))<DatosX[j]])
        X=[X[j] for j in range(Nbin) if j not in Indices_malos]
        Y=[Y[j] for j in range(Nbin) if j not in Indices_malos]
    elif  TipoBin=='Lin':
        bins = np.linspace(MIN, MAX, Nbin)
        X=bins
        for i in range(Nbin):
            if 0<i<Nbin-1:
                Conjunto=[DatosY[j] for j in range(N) if  np.mean(bins[i-1:i+1])<DatosX[j]<=np.mean(bins[i:i+2])]
                if len(Conjunto)==0:
                    Indices_malos+=[i]
                else:
                    Y[i]=np.mean(Conjunto)
                Y[i]=np.mean([DatosY[j] for j in range(N) if  np.mean(bins[i-1:i+1])<DatosX[j]<=np.mean(bins[i:i+2])])
            elif i==0:
                Y[i]=np.mean([DatosY[j] for j in range(N) if  DatosX[j]<=np.mean(bins[i:i+2])])
            else:
                Y[i]=np.mean([DatosY[j] for j in range(N) if  np.mean(bins[i-1:i+1])<DatosX[j]])
        X=[X[j] for j in range(Nbin) if j not in Indices_malos]
        Y=[Y[j] for j in range(Nbin) if j not in Indices_malos]
    else:
        print('TipoBin no valido')
    if TipoBin in ['Log','Lin']:
        return X,Y

def Densidad_Metrica_Regresiones(Regresion,texto,RED,Rangos,metrica,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff,VdeV,Bineo,TipoBin,Porcentaje):

    X=['Red Multiplex','Capa Gramatica','Capa Fonetica']
    Z=['Multi','Gram','Fon']
    Texto=texto+Z[X.index(RED)]
    T_0 = datetime.now()
    S1=Split_gen(Rangos,',')
    R=Split_gen(Rangos,'_')
    print('[Regre Dens][INICIO] '+metrica+' de la '+RED+' con rangos '+S1+' del texto '+texto)
    print('[Regre Dens][INICIO]_Fecha:',datetime.now())
    #Texto=Etiquetas_aleatorio(texto,RED,Seed_Sample,Seed_Shuff,Cant_total_palabras,Cant_muestra_palabras,idioma)[0]
    W1=['Grados','CentrBet']
    W2=['Grados','CentralidadB']
    Etiq=W2[W1.index(metrica)]
    DATOS=dict()
    DATOS['ValoresX']=dict()
    DATOS['Regresion']=dict()
    DATOS['Alfa']=dict()
    if Bineo==False:
        Bin=''
        def  Opcion_bineo(Val_x,Val_y,Rango):
            return Val_x,Val_y
    else:
        Bin='B'+TipoBin+'P'+str(Porcentaje)
        def  Opcion_bineo(Val_x,Val_y,Rango):
            F1,F2=Binear(Val_x,Val_y,TipoBin,Porcentaje)
            F3=F1,F2
            pickle.dump(F3, open(Texto+'R'+str(Rango)+'Dens'+metrica+Bin+".p", "wb" ), protocol=4)
            return F1,F2
    if Regresion=='Ml':
        for Rango in Rangos:
            print('>Cargando datos de la distancia:'+str(Rango))
            T_r=datetime.now()
            print('__Fecha:',T_r)
            D=pickle.load(open(Texto+'R'+str(Rango)+'Dens'+metrica+ ".p", "rb"  ))
            X0=list(D.keys())
            D1=pickle.load(open(Texto+'R'+str(Rango)+metrica+ ".p", "rb"  ))
            X1=[t for t in X0 if t!=0 if VdeV[Rangos.index(Rango)][0]<np.log(t)<VdeV[Rangos.index(Rango)][1]]
            X2=sorted(X1)
            Y2=[D[t] for t in X2]
            X2,Y2=Opcion_bineo(X2,Y2,Rango)
            U=[t for t in D1[Etiq] if t in X1]
            X3=np.log(X2)
            Y3=np.log(Y2)
            X4=X3.reshape(-1,1)
            Y4=Y3.reshape(-1,1)
            model = lm.fit(X4,Y4)
            predictions = lm.predict(X4)
            Y5=np.exp(predictions)
            AlfaMl=lm.coef_[0][0]
            DATOS['ValoresX'][Rango]=X2
            DATOS['Regresion'][Rango]=Y5
            DATOS['Alfa'][Rango]=-AlfaMl
            print('>>Datos de la distancia:'+str(Rango)+' cargados')
            print('__Fecha:')
            print(datetime.now())
            print('Tiempo utilizado:')
            tiempo_desglosado(datetime.now()-T_r)
    elif Regresion=='Mv':
        for Rango in Rangos:
            print('>Cargando datos de la distancia:'+str(Rango))
            T_r=datetime.now()
            print('__Fecha:',T_r)
            D=pickle.load(open(Texto+'R'+str(Rango)+'Dens'+metrica+ ".p", "rb"  ))
            X0=list(D.keys())
            D1=pickle.load(open(Texto+'R'+str(Rango)+metrica+ ".p", "rb"  ))
            X1=[t for t in X0 if t!=0 if VdeV[Rangos.index(Rango)][0]<np.log(t)<VdeV[Rangos.index(Rango)][1]]
            X2=sorted(X1)
            Y2=[D[t] for t in X2]
            X2,Y2=Opcion_bineo(X2,Y2,Rango)
            U=[t for t in D1[Etiq] if t in X1]
            AlfaMv=Formula_Coeficiente_Exp_MaximaV(U)
            C_0=Formula_Coeficiente_Factor_MaximaV(X2,Y2,AlfaMv)
            PredYMv=Aprox_linal_Newman(X2,Y2,AlfaMv)
            DATOS['ValoresX'][Rango]=X2
            DATOS['Regresion'][Rango]=PredYMv
            DATOS['Alfa'][Rango]=AlfaMv
            print('>>Datos de la distancia:'+str(Rango)+' cargados')
            print('__Fecha:')
            print(datetime.now())
            print('Tiempo utilizado:')
            tiempo_desglosado(datetime.now()-T_r)
    if Regresion=='Ml' or Regresion=='Mv':
        pickle.dump(DATOS, open(Texto+'R'+R+'Dens'+metrica+'Regre'+Regresion+Bin+".p", "wb" ), protocol=4)
        print('-Regresiones '+Regresion+' de densidad de '+metrica+' de '+Texto+' con rangos '+S1+' guardadas en:')
        print(Texto+'R'+R+'Dens'+metrica+'Regre'+Regresion+Bin+".p")
        print('[Regre Dens][FIN] '+metrica+' de la '+RED+' con rangos '+S1+' del texto '+texto+' finalizada en:')
        tiempo_desglosado(datetime.now()-T_0)
        print('[Regre Dens][FIN] '+metrica+' Fecha:',datetime.now())
    else:
        print('Regresion no valida')
    return "Densidad_Metrica_Regresiones"

@ray.remote
def Distribuciones_de_grado(texto,RED,Rangos):
    try:
        T_0 = datetime.now()
        if len(Rangos)==1:
            S1=str(Rangos[0])
        else:
            S1=','.join([str(elem) for elem in Rangos])
        print('[Distr][INICIO] Creando distribucion de grado de la '+RED+' con rangos '+S1+' del texto '+texto)
        print('[Distr]_Fecha:',datetime.now())
        X=['Red Multiplex','Capa Gramatica','Capa Fonetica']
        Z=['Multi','Gram','Fon']
        Texto=texto+Z[X.index(RED)]
        for Rango in Rangos:
            print('>>Cargando datos de la distancia:'+str(Rango))
            T_r=datetime.now()
            print('__Fecha:',T_r)
            C=pickle.load(open(Texto+'R'+str(Rango)+'DensGrados'+ ".p", "rb"  ))
            dictionary=dict(C)
            D={}
            F=sorted(dictionary)
            for j in F:
                D[j]=0
                for m in np.arange(F.index(j)+1):
                    D[j]+=dictionary[F[m]]
            pickle.dump(D, open(Texto+'R'+str(Rango)+'DistrGrados'+ ".p", "wb" ), protocol=4)
            print('->Distibucion de grados de la red '+Texto+'R'+str(Rango)+' guardada en:')
            print(Texto+'R'+str(Rango)+'DistrGrados'+ ".p")
            print('__Fecha de conclusion rango '+str(Rango),datetime.now())
            print('->Terminado en:')
            tiempo_desglosado(datetime.now()-T_r)
        print('[Distr][FIN] Proceso '+RED+' con rangos '+S1+' del texto '+texto +' terminado en:')
        tiempo_desglosado(datetime.now()-T_0)
        print('[Distr][FIN]_Fecha:',datetime.now())
        return "Distribuciones_de_grado"
    except Exception as e:
        return "Distribuciones_de_grado-->" + str(e)

@ray.remote
def Clustering_average(texto,RED,Rangos,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff):
    try:
        T_0 = datetime.now()
        S1=Split_gen(Rangos,',')
        R=Split_gen(Rangos,'_')
        print('[Clustering_average][INICIO] Clustering average de la '+RED+' con rangos '+S1+' del texto '+texto)
        print('[Clustering_average][INICIO]_Fecha:',datetime.now())
        Texto=Etiquetas_aleatorio(texto,RED,Seed_Sample,Seed_Shuff,Cant_total_palabras,Cant_muestra_palabras,idioma)[0]
        DATOS=dict()
        for Rango in Rangos:
            print('>Cargando datos de la distancia:'+str(Rango))
            T_r=datetime.now()
            print('__Fecha:',T_r)
            D=pickle.load(open(Texto+'R'+str(Rango)+'Clust'+".p", "rb"  ))
            DATOS[Rango]=np.mean(D.Clustering)
            print('>>Datos del rango:'+str(Rango)+' cargados')
            print('__Fecha:')
            print(datetime.now())
            print('Tiempo utilizado:')
            tiempo_desglosado(datetime.now()-T_r)
        pickle.dump(DATOS, open(Texto+'R'+R+'Clust_Aver'+".p", "wb" ), protocol=4)
        print('-Clustering de la '+RED+' de '+Texto+' con rangos '+S1+' guardadas en:')
        print(Texto+'R'+R+'Clust_Aver'+".p")
        print('[Clustering_average][FIN] Clustering average de la '+RED+' con rangos '+S1+' del texto '+texto+' finalizada en:')
        tiempo_desglosado(datetime.now()-T_0)
        print('[Clustering_average][FIN]  Clustering average Fecha:',datetime.now())
        return "Clustering_average"
    except Exception as e:
        return "Clustering_average-->" + str(e)

@ray.remote
def GradoVsMedida(texto,RED,Seed_Sample,Seed_Shuff,Rangos,Medida,MEDIA,LogX,LogY,Bineo,TipoBin,Porcentaje,Ajuste,Tabla,N,Cant_total_palabras):
    try:
        T_0 = datetime.now()
        if len(Rangos)==1:
             S1=str(Rangos[0])
             R=str(Rangos[0])
        else:
            S1=','.join([str(elem) for elem in Rangos])
            R='_'.join([str(elem) for elem in Rangos])
        if Seed_Sample==0:
            Texto=texto
            TituloT='Texto '+texto
            Muestra=' de las '+str(Cant_total_palabras)+' palabras.'
        else:
            Muestra=' de una muestra de '+str(N)+' palabras de las '+str(Cant_total_palabras)+'.'
            if Seed_Shuff==0:
                Texto=texto+'Seed'+str(Seed_Sample)
                TituloT='Texto '+texto+' semilla sample '+str(Seed_Sample)
            else:
                Texto=texto+'Seed'+str(Seed_Sample)+'Shuf_Seed'+str(Seed_Shuff)
                TituloT='Texto '+texto+' semilla sample '+str(Seed_Sample)+' semilla shuffle '+str(Seed_Shuff)
        X=['Red Multiplex','Capa Gramatica','Capa Fonetica']
        Z=['Multi','Gram','Fon']
        Texto=Texto+Z[X.index(RED)]
        if Bineo==False:
            Bin=''
        else:
            Bin='B'+TipoBin+'P'+str(Porcentaje)
        MEDIDA=['Betweenness','Clustering','AND']
        AbrevMe=['CentrBet','Clust','AND']
        NomMedida=['CentralidadB','Clustering','AND']
        print('[INICIO] Creando comparativa Grado vs '+Medida+' con rangos '+S1+' del texto '+Texto+' de la '+RED)
        print('__Fecha:',datetime.now())
        if Tabla==True:
            DatosTabla=pd.DataFrame(columns=['Rango','Media Grado','Media '+Medida,'Media '+Medida+'/'+'Media Grado'])
            MG=[]
            MM=[]
        if MEDIA==False:
            Y1_lim=[]
        for Rango in Rangos[::-1]:
            print('>Cargando datos de la distancia:'+str(Rango)+' y la '+RED)
            T_r=datetime.now()
            print('__Fecha:',T_r)
            D0=pickle.load(open(Texto+'R'+str(Rango)+ "Grados.p", "rb"  ))
            D=pickle.load(open(Texto+'R'+str(Rango)+AbrevMe[MEDIDA.index(Medida)]+ ".p", "rb"  ))
            if Tabla==True:
                MG+=[np.mean(list(D0['Grados']))]
                MM+=[np.mean(list(D[NomMedida[MEDIDA.index(Medida)]]))]
            else:
                if MEDIA==False:
                    plt.scatter(list(D0['Grados']), list(D[NomMedida[MEDIDA.index(Medida)]]),s=5,color=COLORES2[(Rango-1)%5], label = 'Rango '+str(Rango),alpha=0.5+0.5*(Rango/len(Rangos)))
                    Y1_lim+=[min(list(D[NomMedida[MEDIDA.index(Medida)]])),max(list(D[NomMedida[MEDIDA.index(Medida)]]))]
                else:
                    DF=pickle.load(open(Texto+'R'+str(Rango)+'Media_'+AbrevMe[MEDIDA.index(Medida)]+".p", "rb"  ))
                    DatosX=DF['Grados']
                    DatosY=DF[Medida]
                    Gra0,Med0=Binear(DatosX,DatosY,TipoBin,Porcentaje)
                    plt.plot(Gra0,Med0,markersize=5,color=COLORES2[(Rango-1)%5],label = 'Rango '+str(Rango))
                    if Ajuste==True and Medida=='AND':
                        X3=np.log(Gra0[1:])
                        Y3=np.log(Med0[1:])
                        X4=X3.reshape(-1,1)
                        Y4=Y3.reshape(-1,1)
                        model = lm.fit(X4,Y4)
                        predictions = lm.predict(X4)
                        X5=np.exp(X4)
                        Y5=np.exp(predictions)
                        Alfa2=lm.coef_[0][0]
                        plt.plot(Gra0[1:],Y5,linestyle=ESTILO[(Rango-1)%4],color=COLORES3[0],label='Lm R.'+str(Rango)+':- %.3f'%Alfa2)
            print('>>Datos de la distancia:'+str(Rango)+' de la red '+RED+' cargados')
            print('__Fecha:',datetime.now())
            print('Tiempo utilizado:')
            tiempo_desglosado(datetime.now()-T_r)
        if Tabla==True:
            DatosTabla['Rango']=Rangos[::-1]
            DatosTabla['Media Grado']=MG
            DatosTabla['Media '+Medida]=MM
            DatosTabla['Media '+Medida+'/'+'Media Grado']=np.array(MM)/np.array(MG)
            DatosTabla.to_csv(Texto+'Tabla'+AbrevMe[MEDIDA.index(Medida)]+'R'+R+'.csv', index=False)
            print(DatosTabla)
            print('[FINAL] Tabla comparativa Grado vs '+Medida+' con rangos '+S1+' del texto '+Texto+' de la '+RED+' guardada en:')
            print(Texto+'Tabla'+AbrevMe[MEDIDA.index(Medida)]+'R'+R+'.csv')
        else:
            if MEDIA==False:
                plt.ylim(min(Y1_lim)-(max(Y1_lim)-min(Y1_lim))/20,max(Y1_lim)+(max(Y1_lim)-min(Y1_lim))/20)
            if LogX==True:
                plt.xscale('log')
            if LogY==True:
                plt.yscale('log')
            #plt.grid()
            plt.xlabel('Grado(P)')
            if MEDIA==False:
                plt.ylabel(Medida+'(P)')
                plt.legend(loc='best',fontsize=  'x-small')
                plt.title(RED+' '+Bin)
                plt.suptitle(TituloT+'''
        '''+'Comparacion entre Grado y '+Medida+' por palabras (P), con '+'''
        rangos:'''+','.join([str(elem) for elem in Rangos])+Muestra,y=0.15)
                Nombre_save=Texto+'GradoVs'+AbrevMe[MEDIDA.index(Medida)]+'R'+R
            else:
                plt.ylabel('Media({'+Medida+'(Q):Grado(Q)=Grado(P)})')
                plt.legend(loc='best',fontsize=  'x-small')
                plt.title(RED+' '+Bin)
                plt.suptitle(TituloT+'''
        '''+'Comparacion entre Grado y '+Medida+' por palabras (P), con '+'''
        rangos:'''+','.join([str(elem) for elem in Rangos])+Muestra,y=0.15)
                Nombre_save=Texto+'GradoVs'+AbrevMe[MEDIDA.index(Medida)]+'MediaR'+R
            plt.subplots_adjust(bottom=0.25,top=0.92,left=0.2)
            if LogX==True and LogY==True:
                Save_Final=Nombre_save+'Log'
            elif LogX==True and LogY==False:
                Save_Final=Nombre_save+'LogX'
            elif LogX==False and LogY==True:
                Save_Final=Nombre_save+'LogY'
            else:
                Save_Final=Nombre_save
            if MEDIA==True and Bineo==True:
                Save_Final2=Save_Final+Bin
            else:
                Save_Final2=Save_Final
            if Ajuste==True and Medida=='AND':
                Save_Final2=Save_Final2+'Ajust'
            print('[FINAL] Grafica comparativa Grado vs '+Medida+' con rangos '+S1+' del texto '+Texto+' de la '+RED+' guardada en:')
            print(Save_Final2+'.png')
            plt.savefig(Save_Final2+'.png',dpi=400)
            plt.show()
            plt.clf()
            plt.close()
        print('__Fecha:',datetime.now())
        print('[--] Tiempo utilizado para la comparativa con rangos'+S1)
        tiempo_desglosado(datetime.now()-T_0)
        return "GradoVsMedida"
    except Exception as e:
        return "GradoVsMedida-->" + str(e)

@ray.remote
def GradoVsMedia_save(texto,RED,Seed_Sample,Seed_Shuff,Rangos,Medida):
    try:
        T_0 = datetime.now()
        if len(Rangos)==1:
             S1=str(Rangos[0])
        else:
            S1=','.join([str(elem) for elem in Rangos])
        if Seed_Sample==0:
            Texto=texto
        else:
            if Seed_Shuff==0:
                Texto=texto+'Seed'+str(Seed_Sample)
            else:
                Texto=texto+'Seed'+str(Seed_Sample)+'Shuf_Seed'+str(Seed_Shuff)
        X=['Red Multiplex','Capa Gramatica','Capa Fonetica']
        Z=['Multi','Gram','Fon']
        Texto=Texto+Z[X.index(RED)]
        MEDIDA=['Betweenness','Clustering','AND']
        AbrevMe=['CentrBet','Clust','AND']
        NomMedida=['CentralidadB','Clustering','AND']
        print('[INICIO] Iniciando de obtencion de Media'+Medida+' con rangos '+S1+' del texto '+Texto+' de la '+RED)
        print('__Fecha:',datetime.now())
        for Rango in Rangos:
            print('>Cargando datos de la distancia:'+str(Rango)+' y la '+RED)
            T_r=datetime.now()
            print('__Fecha:',T_r)
            D0=pickle.load(open(Texto+'R'+str(Rango)+ "Grados.p", "rb"  ))
            D=pickle.load(open(Texto+'R'+str(Rango)+AbrevMe[MEDIDA.index(Medida)]+ ".p", "rb"  ))
            Gra0=list(set(D0['Grados']))
            Med0=[]
            for k in set(D0['Grados']):
                PALABRAS=[P for P in list(D0['Grados'].keys()) if D0['Grados'][P]==k]
                VALORES=[D[NomMedida[MEDIDA.index(Medida)]][j] for j in PALABRAS]
                Med0+=[np.array(VALORES).mean()]
            DF={}
            DF['Grados']=Gra0
            DF[Medida]=Med0
            pickle.dump(DF, open(Texto+'R'+str(Rango)+'Media_'+AbrevMe[MEDIDA.index(Medida)]+".p", "wb" ), protocol=4)
            print('-Media Clustering de la '+RED+' con rango '+str(Rango)+' guardada en:')
            print(Texto+'R'+str(Rango)+'Media_'+AbrevMe[MEDIDA.index(Medida)]+".p")
            print('__Fecha de la obteniencion de la media de clustering de la red '+RED+' con rango '+str(Rango),datetime.now())
            print('->Terminado en:')
            tiempo_desglosado(datetime.now()-T_r)
        print('[FINAL] Datos Grado vs Media '+Medida+' con rangos '+S1+' del texto '+Texto+' de la '+RED+' guardados.')
        print('__Fecha:',datetime.now())
        print('[--] Tiempo utilizado para la comparativa con rangos'+S1)
        tiempo_desglosado(datetime.now()-T_0)
        return "GradoVsMedia_save"
    except Exception as e:
        return "GradoVsMedia_save-->" + str(e)

@ray.remote
def Separar_comunidades(texto,RED,Seed_Sample,Seed_Shuff,Rangos):
    try:
        T_0 = datetime.now()
        if len(Rangos)==1:
             S1=str(Rangos[0])
             R=str(Rangos[0])
        else:
            S1=','.join([str(elem) for elem in Rangos])
            R='_'.join([str(elem) for elem in Rangos])
        if Seed_Sample==0:
            Texto=texto
        else:
            if Seed_Shuff==0:
                Texto=texto+'Seed'+str(Seed_Sample)
            else:
                Texto=texto+'Seed'+str(Seed_Sample)+'Shuf_Seed'+str(Seed_Shuff)
        print('[INICIO][Comunidades Louvain] Creando la mejor particion de la red con rangos '+S1+' del texto '+Texto+' de la '+RED)
        print('[INICIO][Comunidades Louvain]Fecha:',datetime.now())
        X=['Capa Gramatica','Capa Fonetica','Red Multiplex']
        Z=['Gram','Fon','Multi']
        Texto=Texto+Z[X.index(RED)]
        for Rango in Rangos:
            T_r=datetime.now()
            G=pickle.load(open(Texto+'R'+str(Rango)+ "Red.p", "rb"  ))
            partition = community_louvain.best_partition(G)
            pickle.dump(partition, open(Texto+'R'+str(Rango)+"Part_Louvain.p", "wb" ), protocol=4)
            print(' Mejor particion de la red con rango '+str(Rango)+' del texto '+Texto+' de la '+RED+' guardada en:')
            print(Texto+'R'+str(Rango)+"Part_Louvain.p")
            print('->Tiempo utilizado en el rango '+str(Rango)+':')
            tiempo_desglosado(datetime.now()-T_r)
        print('[FIN][Comunidades Louvain]->Proceso completo de los rangos '+S1+' terminado en:')
        tiempo_desglosado(datetime.now()-T_0)
        return "Separar_comunidades"
    except Exception as e:
        return "Separar_comunidades-->" + str(e)

@ray.remote
def Modularidad(texto,Rangos):
    try:
        T_0 = datetime.now()
        if len(Rangos)==1:
             S1=str(Rangos[0])
             R=str(Rangos[0])
        else:
            S1=','.join([str(elem) for elem in Rangos])
            R='_'.join([str(elem) for elem in Rangos])
        print('[INICIO][MODULARIDAD] Construyendo  tabla comparativa modularidad con rangos '+S1+' del texto '+texto)
        print('[INICIO][MODULARIDAD]Fecha:',datetime.now())
        DatosTabla=pd.DataFrame(columns=['Rango','Modularidad Gramatica','Modularidad Fonetica','Modularidad Multiplex'])
        M_gram=[]
        M_fon=[]
        M_multi=[]
        for Rango in Rangos:
            T_r=datetime.now()
            G=pickle.load(open(texto+'Gram'+'R'+str(Rango)+ "Red.p", "rb"  ))
            partition=pickle.load(open(texto+'Gram'+'R'+str(Rango)+"Part_Louvain.p", "rb"  ))
            M_gram+=[community_louvain.modularity(partition, G)]
            G=pickle.load(open(texto+'Fon'+'R'+str(Rango)+ "Red.p", "rb"  ))
            partition=pickle.load(open(texto+'Fon'+'R'+str(Rango)+"Part_Louvain.p", "rb"  ))
            M_fon+=[community_louvain.modularity(partition, G)]
            G=pickle.load(open(texto+'Multi'+'R'+str(Rango)+ "Red.p", "rb"  ))
            partition=pickle.load(open(texto+'Multi'+'R'+str(Rango)+"Part_Louvain.p", "rb"  ))
            M_multi+=[community_louvain.modularity(partition, G)]
            print('->Tiempo utilizado en el rango '+str(Rango)+':')
            tiempo_desglosado(datetime.now()-T_r)
        DatosTabla['Rango']=Rangos
        DatosTabla['Modularidad Gramatica']=M_gram
        DatosTabla['Modularidad Fonetica']=M_fon
        DatosTabla['Modularidad Multiplex']=M_multi
        DatosTabla.to_csv(texto+'TablaModularidad'+'R'+R+'.csv', index=False)
        print(DatosTabla)
        print('[FINAL][MODULARIDAD] Tabla comparativa modularidad con rangos '+S1+' del texto '+texto+' guardada en:')
        print(texto+'TablaModularidad'+'R'+R+'.csv')
        print('[FINAL][MODULARIDAD] Proceso completo de los rangos '+S1+' terminado en:')
        tiempo_desglosado(datetime.now()-T_0)
        return "Modularidad"
    except Exception as e:
        return "Modularidad-->" + str(e)

def Distancia_Jensen_generalizado(Dens_prob_P,Dens_prob_Q,base=None):
    A=set(Dens_prob_P.keys())
    B=set(Dens_prob_Q.keys())
    C=A.union(B)
    Px=[]
    Qx=[]
    for c in C:
        if c in A:
            Px+=[Dens_prob_P[c]]
        else:
            Px+=[0]
    for c in C:
        if c in B:
            Qx+=[Dens_prob_Q[c]]
        else:
            Qx+=[0]

    if base is not None:
        Distancia_J=jensenshannon(Px,Qx,base)
    else:
        Distancia_J=jensenshannon(Px,Qx)
    return Distancia_J

@ray.remote
def Comparacion_Jensen(texto,Rangos,Dist_o_Div,base=None):
    try:
        T_0=datetime.now()
        if len(Rangos)==1:
             S1=str(Rangos[0])
             R=str(Rangos[0])
        else:
            S1=','.join([str(elem) for elem in Rangos])
            R='_'.join([str(elem) for elem in Rangos])
        Y=['Capa Fonetica','Capa Gramatica']
        X=['Capa Gramatica','Capa Fonetica']
        Z1=['Gram','Fon']
        Z2=['Fon','Gram']
        Min_Jensen=[]
        Max_Jensen=[]
        print('[INICIO][Comparativa Jensen] Fecha '+texto+' con rangos '+S1+':',datetime.now())
        if Dist_o_Div=='Distancia':
            Etiqueta_safe='DistJensen'
        else:
            Etiqueta_safe='DivJensen'
        for i in Y:
            for j in X:
                T_r=datetime.now()
                Datos=dict()
                for r1 in Rangos:
                    d=[]
                    C1=pickle.load(open(texto+Z2[Y.index(i)]+'R'+str(r1)+'DensGrados'+ ".p", "rb"  ))
                    for r2 in Rangos:
                        C2=pickle.load(open(texto+Z1[X.index(j)]+'R'+str(r2)+'DensGrados'+ ".p", "rb"  ))
                        if Dist_o_Div=='Distancia':
                            d+=[Distancia_Jensen_generalizado(C1,C2,base)]
                        else:
                            d+=[(Distancia_Jensen_generalizado(C1,C2,base))**2]
                    Datos[r1]=d
                DF=pd.DataFrame(data=Datos,index=Rangos)
                Min_Jensen+=[DF.min().min()]
                Max_Jensen+=[DF.max().max()]
                pickle.dump(DF, open(texto+'R'+R+Etiqueta_safe+Z2[Y.index(i)]+'VS'+Z1[X.index(j)]+".p", "wb"), protocol=4)
                print('Comparacion Jensen '+i+'  con '+j+' del texto '+texto+' con rangos '+S1+' guardada en:')
                print(texto+'R'+R+Etiqueta_safe+Z2[Y.index(i)]+'VS'+Z1[X.index(j)]+".p")
                print('->Tiempo utilizado para comparar '+i+'  con '+j+':')
                tiempo_desglosado(datetime.now()-T_r)
        Extremos_Jensen=min(Min_Jensen),max(Max_Jensen)
        pickle.dump(Extremos_Jensen, open(texto+'R'+R+Etiqueta_safe+"Extremos.p", "wb" ), protocol=4)
        print('Valores extremos de Comparacion Jensen del texto '+texto+' con rangos '+S1+' guardada en:')
        print(texto+'R'+R+Etiqueta_safe+"Extremos.p")
        print('min='+str(Extremos_Jensen[0]))
        print('max='+str(Extremos_Jensen[1]))
        print('[FIN][Comparativa Jensen]_Fecha:',datetime.now())
        print('[FIN][Comparativa Jensen] Tiempo utilizado para la comparativa con rangos'+S1)
        tiempo_desglosado(datetime.now()-T_0)
        return "Comparacion_Jensen"
    except Exception as e:
        return "Comparacion_Jensen-->" + str(e)

#################################################
#                 Medidas

#idioma in IDIOMAS                                               , recordar no hacer para español
#N in [1,...,Cant_total_palabras]                                , se sugiere N                  = 10000
#Cant_total_palabras in [1,...]                                  , se tomara  Cant_total_palabras= Textos_x_idioma
#Seed_Sample in [1,...,Cant_total_palabras]                      , se sugiere Seed_Sample        = 5                                                 , para  que concuerde con el de español
#Seed_Shuff in [1,...,Cant_total_palabra                         , se sugiere Seed_Shuff         = 0                                                 , porque no se  ha hecho un intercambio interno de caracteres
#Rangos=[n1,n2,...]                                              , se sugiere Rangos             = [1,2,3,4,5]                                       , pero puede ser cualquier lista en el orden que sea
#METRICAS sublist ['Grados','CentrBet','Clust','AND']            , se sugiere METRICAS           = ['Grados','CentrBet','Clust','AND']
#VdeV [[f(i),g(i)] for i in Rangos]                              , se sugiere VdeV               = [[-10**10,10**10] for i in Rangos]                , por CentrBet que cae en [0,1]
#Bineo in [True,False]                                           , se sugiere Bineo              = True,False
#TipoBin in ['Log','Lin']                                        , se sugiere TipoBin            = 'Log'
#Porcentaje in [1,...,100]                                       , se sugiere Porcentaje         = 10                                                , es el porcentaje de BINS
#Regresion in ['Ml','Mv']                                        , se sugiere hacer las dos, ya esta en el codigo
#Redes sublist ['Capa Gramatica','Capa Fonetica','Red Multiplex'], se sugiere Redes              = ['Capa Gramatica','Capa Fonetica','Red Multiplex']
#Medidas sublist ['Betweenness','Clustering','AND']              , se sugiere Medidas            = ['Betweenness','Clustering','AND']
#Dist_o_Div in ['Distancia','Divergencia']                       , se sugiere hacer las dos, ya esta en el codigo

def Medidas_x_idioma(idioma,N,Cant_total_palabras,Seed_Sample,Seed_Shuff,Rangos,METRICAS,VdeV,Bineo,TipoBin,Porcentaje,Redes,Medidas):
    T_0 = datetime.now()
    print('[INICIO][Medidas] Iniciando a conseguir  las medidas por idioma '+',idioma='+str(idioma)+',N='+str(N)+',Cant_total_palabras='+str()+',Seed_Sample='+str(Seed_Sample)+',Seed_Shuff='+str(Seed_Shuff)+',Rangos='+str(Rangos)+',METRICAS='+str(METRICAS)+',VdeV='+str(VdeV)+',Bineo='+str(Bineo)+',TipoBin='+str(TipoBin)+',Porcentaje='+str(Porcentaje)+',Redes='+str(Redes)+',Medidas='+str(Medidas))
    print('Fecha:',datetime.now())
    Cant_muestra_palabras=N
    texto=str(len(TEXTOS[idioma]))+'Txt'+idioma+'N'+'Sample'+str(N)
    texto2=Etiquetas_aleatorio(texto,'Otra Cosa',Seed_Sample,Seed_Shuff,Cant_total_palabras,Cant_muestra_palabras,idioma)[0]
    result_ids =[]
    for RED in Redes:
        for metrica in METRICAS:
            if metrica == 'CentrBet':
                MetricaBet(texto2,RED,Rangos,metrica,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff,VdeV,Bineo,TipoBin,Porcentaje)
        ray.init()
        for metrica in METRICAS:
            if metrica != 'CentrBet':
                Metrica(texto2,RED,Rangos,metrica,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff,VdeV,Bineo,TipoBin,Porcentaje)
        result_ids = []
        result_ids.append(Distribuciones_de_grado.remote(texto2,RED,Rangos))
        result_ids.append(Clustering_average.remote(texto,RED,Rangos,Cant_total_palabras,Cant_muestra_palabras,idioma,Seed_Sample,Seed_Shuff))
        for Medida in Medidas:
            result_ids.append(GradoVsMedia_save.remote(texto,RED,Seed_Sample,Seed_Shuff,Rangos,Medida))
            result_ids.append(GradoVsMedida.remote(texto,RED,Seed_Sample,Seed_Shuff,Rangos,Medida,True,True,True,True,'Log',Porcentaje,True,True,N,Cant_total_palabras))
        result_ids.append(Separar_comunidades.remote(texto,RED,Seed_Sample,Seed_Shuff,Rangos))
        results = ray.get(result_ids)  # [0, 1, 2, 3]
        ray.shutdown()
        print(results)
    result_ids = []
    ray.init()
    result_ids.append(Modularidad.remote(texto2,Rangos))
    for Dist_o_Div in ['Distancia','Divergencia']:
         result_ids.append(Comparacion_Jensen.remote(texto2,Rangos,Dist_o_Div))
    results = ray.get(result_ids)  # [0, 1, 2, 3]
    ray.shutdown()
    print(results)
    print('[FIN][Medidas] Medidas calculadas por idioma '+',idioma='+str(idioma)+',N='+str(N)+',Cant_total_palabras='+str()+',Seed_Sample='+str(Seed_Sample)+',Seed_Shuff='+str(Seed_Shuff)+',Rangos='+str(Rangos)+',METRICAS='+str(METRICAS)+',VdeV='+str(VdeV)+',Bineo='+str(Bineo)+',TipoBin='+str(TipoBin)+',Porcentaje='+str(Porcentaje)+',Redes='+str(Redes)+',Medidas='+str(Medidas)+' en:')
    tiempo_desglosado(datetime.now()-T_0)
    print('[FIN][Medidas] Fecha:',datetime.now())

#################################################################
################           JUNTO              ###################
#################################################################



#-------------------------------------------------------------------------------
#            Funciones para  generar redes (Textos_x_idioma)
#-------------------------------------------------------------------------------



def Todo_x_idioma(idioma,N,Seed_Sample):
    T_0 = datetime.now()
    print('[INICIO][Todo_Construccion] Iniciando a conseguir  las medidas por idioma '+',idioma='+str(idioma)+',N='+str(N)+',Seed_Sample='+str(Seed_Sample))
    print('Fecha:',datetime.now())
    #variables fijas:
    limitador           = 1
    Minusculas          = True
    Eliminar_Arti_y_prep= True
    Parte_empiezo       = 0
    Parte_final         = N
    Corte               = 2000
    Rangos              = [1,2,3]
    METRICAS            = ['Grados','CentrBet','Clust','AND']
    VdeV                = [[-10**10,10**10] for i in Rangos]
    TipoBin             = 'Log'
    Porcentaje          = 10
    Redes               = ['Red Multiplex']#['Capa Gramatica','Capa Fonetica','Red Multiplex']
    Medidas             = ['Betweenness','Clustering','AND']
    V_Lim_by_R          = [[0,N] for i in Rangos]
    Seed_Shuff          = 0
    Nnodes              = 40
    #Funciones:
    ## Texto:
    Textos_x_idioma(idioma,limitador,Minusculas,Eliminar_Arti_y_prep,N,Seed_Sample)
    Cant_total_palabras =len(pickle.load(open(str(len(TEXTOS[idioma]))+'Txt'+idioma+'N'+".p", "rb"  )))
    ## Redes:
    #Redes_Partes_x_idioma(idioma,N,Seed_Sample,Parte_empiezo,Parte_final,Corte,Rangos,Nnodes)
    #Redes_x_idioma(idioma,N,Seed_Sample,V_Lim_by_R,Rangos)
    #Red_Multiplex_x_idioma(idioma,N,Seed_Sample,Seed_Shuff,Rangos)
    ## medidas:
    for Bineo in [True,False]:
        Medidas_x_idioma(idioma,N,Cant_total_palabras,Seed_Sample,Seed_Shuff,Rangos,METRICAS,VdeV,Bineo,TipoBin,Porcentaje,Redes,Medidas)
    print('[FIN][Todo_Construccion] Textosconstruidos por idioma '+',idioma='+str(idioma)+',N='+str(N)+',Seed_Sample='+str(Seed_Sample)+' en:')
    tiempo_desglosado(datetime.now()-T_0)
    print('[FIN][Todo_Construccion]Fecha:',datetime.now())

Todo_x_idioma('Español',10000,5)
