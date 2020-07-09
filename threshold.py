# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:21:07 2020

@author: BALAMLAPTOP2
Input variables: 
    type_proj = ['icd', 'atc']
"""

import networkx as nx
from networkx.algorithms import bipartite
#import matplotlib.pyplot as plt
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
            print(index_source)
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
    print("Saving values for the given threshold ...")
    if type_proj == 0:
        with open("threshold_icd.txt", "a+") as f:
            f.write(str(len(components))+","+str(nodes_connected)+","+str(nodes_unconnected)+","+str(lcs)+","+str(avg_degree)+"\n")
        nx.write_graphml(H,'ICD/projICD_th_'+str(th)+'.graphml')
    elif type_proj == 1:
        with open("threshold_atc.txt", "a+") as f:
            f.write(str(len(components))+","+str(nodes_connected)+","+str(nodes_unconnected)+","+str(lcs)+","+str(avg_degree)+"\n")
        nx.write_graphml(H,'ATC/projATC_th_'+str(th)+'.graphml')
    else:
        print("The option doesn't exist. Try again.")
    

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
    p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-5)
    #timing it...
    start = time.time()
#    for file in names:
#        p.apply_async(multip, [file])
        
    if type_proj == 0:
        for th in sorted(list(counterCIE.keys())):
            p.apply_asinc(threshold_analysis, [C, th, type_proj, len(degY)])
#            threshold_analysis(C, th, type_proj, len(degY))
    elif type_proj == 1:
        for th in sorted(list(counterATC.keys())):
            p.apply_asinc(threshold_analysis, [C, th, type_proj, len(degX)])
#            threshold_analysis(C, th, type_proj, len(degX))

    p.close()
    p.join()
    print("Complete")
    end = time.time()
    print('total time (s)= ' + str(end-start))
    
    
    
    
#    c_list_cie = []
#    nc_list_cie = []
#    nuc_list_cie = []
    
    
    #for th in sorted(list(counterCIE.keys())):
    #th = 2
#    H = nx.Graph()
#    index_source = 1
#    index_target = 1
#    for n in C.nodes(data=True):
#        if n[1]['bipartite'] == 0:
#            print(index_source)
#            sourceNode = n[0]
#            s_neighbors = set(C.neighbors(n[0]))
#            for m in C.nodes(data = True):
#                if m[1]['bipartite'] == 0: #### Change to 1 to change the projection to active ingredient
#                    #print(index_target)
#                    targetNode = m[0]
#                    t_neighbors = set(C.neighbors(m[0]))
#                    if sourceNode != targetNode and index_target > index_source:
#                        if len(s_neighbors & t_neighbors) >= th:
#                            H.add_node(sourceNode)
#                            H.add_node(targetNode)
#                            H.add_edge(sourceNode,targetNode)
#                    index_target += 1
#            index_target = 1
#            index_source += 1
#    
#        
#    components = sorted(nx.connected_components(H), key=len, reverse=True)
#    #sum(list(map(lambda c: len(c), components)))i
#    c_list_cie.append(len(components))
#    nodes_connected = sum(list(map(lambda c: len(c), components)))
#    nc_list_cie.append(nodes_connected)
#    nuc_list_cie.append(len(nodes_0) - nodes_connected)
#    nx.write_graphml(H,'ICD/projICD_th_'+str(th)+'.graphml')
#        
#    fig, ax1 = plt.subplots()
#    #color = 'tab: black'
#    #ax1.set_ylabel('# Components')
#    ax1.set_xlabel('Degree (k)')
#    ax1.plot(sorted(list(counterCIE.keys())),nc_list_cie, color = 'black', label = '# Connected Nodes')
#    ax1.plot(sorted(list(counterCIE.keys())),nuc_list_cie, color = 'r', label = '# Unconnected Nodes')
#    #ax1.tick_params(axis = 'y')
#    plt.legend(loc='center right')
#    ax2 = ax1.twinx()
#    ax2.plot(sorted(list(counterCIE.keys())),c_list_cie, color = 'blue', linestyle=':', label = '# Connected Components')
#    #ax2.tick_params(axis = 'y')
#    ax2.set_ylabel('# Connected Components')
#    #fig.tight_layout()
#    #plt.legend(loc='lower right')
#    #plt.legend(loc='lower right')
#    #plt.show()
#    
#    name = "connected_components_icd_gcs"
#    #plt.savefig(name + '.eps')
#    plt.savefig(name + '.png', dpi = 1000)
#    plt.clf()
#    
#    ##### ACTIVE INGREDIENTS
#    c_list = []
#    nc_list = []
#    nuc_list = []
#    for th in sorted(list(counterATC.keys())):
#        H = nx.Graph()
#        index_source = 1
#        index_target = 1
#        for n in C.nodes(data=True):
#            if n[1]['bipartite'] == 1:
#                sourceNode = n[0]
#                s_neighbors = set(C.neighbors(n[0]))
#                for m in C.nodes(data = True):
#                    if m[1]['bipartite'] == 1: #### Change to 1 to change the projection to active ingredient
#                        targetNode = m[0]
#                        t_neighbors = set(C.neighbors(m[0]))
#                        if sourceNode != targetNode:
#                            if len(s_neighbors & t_neighbors) >= th:
#                                H.add_node(sourceNode)
#                                H.add_node(targetNode)
#                                H.add_edge(sourceNode,targetNode)
#                        index_target += 1
#                index_source += 1
#                                
#        components = sorted(nx.connected_components(H), key=len, reverse=True)
#        c_list.append(len(components))
#        nodes_connected = sum(list(map(lambda c: len(c), components)))
#        nc_list.append(nodes_connected)
#        nuc_list.append(len(nodes_1) - nodes_connected)
#        nx.write_graphml(H,'ATC/projATC_th_'+str(th)+'.graphml')
#        
#        
#    fig, ax1 = plt.subplots()
#    #color = 'tab: black'
#    #ax1.set_ylabel('# Components')
#    #ax1.set_xlabel('Degree')
#    ax1.set_xlabel('Degree (k)')
#    ax1.plot(sorted(list(counterATC.keys())),nc_list, color = 'black', label = '# Connected Nodes')
#    ax1.plot(sorted(list(counterATC.keys())),nuc_list, color = 'r', label = '# Unconnected Nodes')
#    #ax1.tick_params(axis = 'y')
#    plt.legend(loc='center right')
#    ax2 = ax1.twinx()
#    ax2.plot(sorted(list(counterATC.keys())),c_list, color = 'blue', linestyle=':', label = '# Connected Components')
#    ax2.set_ylabel('# Connected Components')
#    
#    name = "connected_components_atc_gcs"
#    #plt.savefig(name + '.eps')
#    plt.savefig(name + '.png', dpi = 1000)
#    plt.clf()