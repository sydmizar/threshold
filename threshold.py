# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:21:07 2020

@author: BALAMLAPTOP2
Input variables: 
    type_proj = ['icd', 'atc']
"""

import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import collections
import pandas as pd
import multiprocessing
import sys
import time

def threshold_analysis(C, th, type_proj, nn):
    print("Creating new graph by given threshold ... "+str(th))
    H = nx.Graph()
    index_source = 1
    index_target = 1
    for n in C.nodes(data=True):
        if n[1]['bipartite'] == type_proj:
            #print(index_source)
            sourceNode = n[0]
            s_neighbors = set(C.neighbors(n[0]))
            for m in C.nodes(data = True):
                if m[1]['bipartite'] == type_proj: #### Change to 1 to change the projection to active ingredient
                    targetNode = m[0]
                    t_neighbors = set(C.neighbors(m[0]))
                    if sourceNode != targetNode and index_target > index_source:
                        if len(s_neighbors & t_neighbors) >= th:
                            H.add_node(sourceNode)
                            H.add_node(targetNode)
                            H.add_edge(sourceNode,targetNode)
                    index_target += 1
            index_target = 1
            index_source += 1        					
    components = sorted(nx.connected_components(H), key=len, reverse=True)
    nodes_connected = sum(list(map(lambda c: len(c), components)))
    nodes_unconnected = nn - nodes_connected
    lcs = len(components[0])
    degrees = H.degree()
    sum_of_edges = sum(list(dict(degrees).values()))
    avg_degree = sum_of_edges / H.number_of_nodes()
    print("Saving values for the given threshold ..."+str(th))
    if type_proj == 0:
        with open("threshold_icd.txt", "a+") as f:
            f.write(str(th)+","+str(len(components))+","+str(nodes_connected)+","+str(nodes_unconnected)+","+str(lcs)+","+str(avg_degree)+"\n")
        nx.write_graphml(H,'ICD/projICD_th_'+str(th)+'.graphml')
    elif type_proj == 1:
        with open("threshold_atc.txt", "a+") as f:
            f.write(str(th)+","+str(len(components))+","+str(nodes_connected)+","+str(nodes_unconnected)+","+str(lcs)+","+str(avg_degree)+"\n")
        nx.write_graphml(H,'ATC/projATC_th_'+str(th)+'.graphml')
    else:
        print("The option doesn't exist. Try again.")
        
def graphic_connected_components(type_proj, degree):
    if type_proj == 0:
        threshold_data = pd.read_csv("threshold_icd.txt", sep = ",", header = None)
    elif type_proj == 1:
        threshold_data = pd.read_csv("threshold_atc.txt", sep = ",", header = None)
    
    threshold_data.columns = ["threshold", "conn_components", "conn_nodes", "unconn_nodes", "lcs", "avg_degree"]
    threshold_data = threshold_data.sort_values('threshold')
    
    n = len(degree) - threshold_data.shape[0]
    
    lstdegree = sorted(list(degree.keys()))[:len(degree) - n]
    
    fig, ax1 = plt.subplots()
    #color = 'tab: black'
    #ax1.set_ylabel('# Components')
    #ax1.set_xlabel('Degree')
    ax1.set_xlabel('Degree (k)')
    ax1.plot(lstdegree, threshold_data['conn_nodes'], color = 'black', label = '# Connected Nodes')
    ax1.plot(lstdegree, threshold_data['unconn_nodes'], color = 'r', label = '# Unconnected Nodes')
    #ax1.tick_params(axis = 'y')
    plt.legend(loc='center right')
    ax2 = ax1.twinx()
    ax2.plot(lstdegree, threshold_data['conn_components'], color = 'blue', linestyle=':', label = '# Connected Components')
    ax2.set_ylabel('# Connected Components')
    
    name = "images/connected_components_"+type_proj
    #plt.savefig(name + '.eps')
    plt.savefig(name + '.png', dpi = 1000)
    plt.clf()

    
    

if __name__ == '__main__':

    type_proj = int(sys.argv[1])
    print("Reading file ...")

    vdmdata = pd.read_csv('vdmdata_reduce.csv', encoding = 'utf-8-sig')
    
    nodes_0 = []
    nodes_1 = []
    for m in vdmdata.iterrows():
        nodes_0.append(m[1][0]) #ICD
        nodes_1.append(m[1][1]) #ATC
        
    nodes_0 = list(dict.fromkeys(nodes_0))
    nodes_1 = list(dict.fromkeys(nodes_1))
    print("Building a bipartite graph ...")
    # Build a bipartite graph:
    G = nx.Graph()
    # Add nodes ATC - ICD
    G.add_nodes_from(nodes_0, bipartite=0) # Add the node attribute “bipartite” disease
    G.add_nodes_from(nodes_1, bipartite=1) # active substance
    
    # Add edges without weight
    for m in vdmdata.iterrows():
        enfermedad = m[1][0];
        #peso = m[1][3];
        sustancia = m[1][1];
        G.add_edge(enfermedad, sustancia)
    
    # Get the largest component
    print("Getting largest component ...")
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_component = components[0]
    C = G.subgraph(largest_component)
        
    degX,degY=bipartite.degrees(C,nodes_0)
    degATC = dict(degX).values()
    degCIE = dict(degY).values()
    counterATC = collections.Counter(degATC)
    counterCIE = collections.Counter(degCIE)
    
    #use one less process to be a little more stable
    #p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-5)
    p = multiprocessing.Pool()
    #timing it...
    start = time.time()
#    for file in names:
#        p.apply_async(multip, [file])
        
    if type_proj == 0:
#        for th in sorted(list(counterCIE.keys())):
#            p = multiprocessing.Process(target=threshold_analysis, args=(C, th, type_proj, len(degY)))
#            p.start()
#            #p.join()
            
#        with multiprocessing.Pool() as p:
#            for th in sorted(list(counterCIE.keys())):
#                p.apply_async(threshold_analysis, [C, th, type_proj, len(degY)])
#            p.map(threshold_analysis)
        for th in sorted(list(counterCIE.keys())):
            p.apply_async(threshold_analysis, [C, th, type_proj, len(degY)])
            
#        graphic_connected_components(type_proj, counterCIE)
#            threshold_analysis(C, th, type_proj, len(degY))
    elif type_proj == 1:
        for th in sorted(list(counterATC.keys())):
            p.apply_async(threshold_analysis, [C, th, type_proj, len(degX)])
#        graphic_connected_components(type_proj, counterATC)
#            threshold_analysis(C, th, type_proj, len(degX))
        
    
            
    

    p.close()
    p.join()
    print("Complete")
    end = time.time()
    print('total time (s)= ' + str(end-start))
    