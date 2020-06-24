# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:07:37 2020

@author: BALAMLAPTOP2
"""

import numpy as np
import networkx as nx
import sys
from networkx.algorithms import bipartite
import pandas as pd

if __name__ == '__main__':
    type_nx = sys.argv[1]
    type_proj = sys.argv[2]
    
    vdmdata_reduce = pd.read_csv('vdmdata_reduce.csv')

    nodes_0 = []
    nodes_1 = []
    for m in vdmdata_reduce.iterrows():
        nodes_0.append(m[1][0]) #ICD
        nodes_1.append(m[1][1]) #ATC
        
    nodes_0 = list(dict.fromkeys(nodes_0))
    nodes_1 = list(dict.fromkeys(nodes_1))
    
    # Build a bipartite graph:
    G = nx.Graph()
    G.add_nodes_from(nodes_0, bipartite=0) # disease
    G.add_nodes_from(nodes_1, bipartite=1) # active substance
    
    
    for m in vdmdata_reduce.iterrows():
        enfermedad = m[1][0];
        sustancia = m[1][1];
        G.add_edge(enfermedad, sustancia)
    
    if type_nx == 'projected' and type_proj == 'icd':
        # Build Projected Graph Diseases
        GP = bipartite.weighted_projected_graph(G, nodes_0)
        print('Calculate Global properties for projected graph '+type_proj)
        print("\n")
    elif type_nx == 'projected' and type_proj == 'atc':
        # Build Projected Graph Active Ingredients
        GP = bipartite.weighted_projected_graph(G, nodes_1)
        print('Calculate Global properties for projected graph '+type_proj)
        print("\n")
    else:
        print('Calculate Global properties for bipartite network')
        print("\n")
    
    if type_nx == 'bipartite':
        print("Nodes Number : "+str(G.number_of_nodes()))
        print("\n")
        print("Edges Number : "+str(G.number_of_edges()))
        print("\n")
        print('Calculating density ...')
        print("\n")
        print("Density ICD Nodes (Diseases): "+str(bipartite.density(G, nodes_0)))
        print("\n")
        print("Density ATC Nodes (Active Substances): "+str(bipartite.density(G, nodes_1)))
        print("\n")
        print('Calculating mean degree ...')
        print("\n")
        G_deg = nx.degree_histogram(G)
        G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
        print('average degree: {}'.format(sum(G_deg_sum) / G.number_of_nodes()))
        print("\n")
        print('Calculating mean clustering ...')
        print("\n")
        cluster_g = bipartite.clustering(G)
        scg = 0;
        for i in range(len(cluster_g)):
            scg = scg + list(cluster_g.items())[i][1]
        print("Average clustering %s" % str(scg/len(cluster_g)))
        print("\n")
    else:
        print("Nodes Number : "+str(GP.number_of_nodes()))
        print("\n")
        print("Edges Number : "+str(GP.number_of_edges()))
        print("\n")
        
        print('Calculating density ...')
        print("\n")
        components = sorted(nx.connected_components(GP), key=len, reverse=True)
        largest_component = components[0]
        C = GP.subgraph(largest_component)
        print("Density Largest Component: %s" % str(nx.density(C)))
        print("\n")
        print("Density projected graph: %s" % str(nx.density(GP)))
        print("\n")
        print('Calculating mean degree ...')
        print("\n")
        G_deg = nx.degree_histogram(GP)
        G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
        print('Average degree: {}'.format(sum(G_deg_sum) / GP.number_of_nodes()))
        print("\n")
        print('Calculating mean clustering ...')
        print("\n")
        print("Average: %s" % str(nx.average_clustering(C)))
        print("\n")
        print('Calculating mean shortest path lenght ...')
        print("\n")
        
        pathlengths = []
        i = 0
        m_pl_icd = np.zeros((GP.number_of_nodes(), GP.number_of_nodes()))
        for v in GP.nodes():
            spl = dict(nx.single_source_shortest_path_length(GP, v))
            sorted_dict = {k: spl[k] for k in sorted(spl)}
            j = 0
            for p in sorted_dict:
                #if v == p:
                pathlengths.append(sorted_dict[p])
                m_pl_icd[i][j] = sorted_dict[p]
                j += 1
            i += 1
            
        if type_proj == 'icd':
            df = pd.DataFrame(m_pl_icd, index=nodes_0, columns=nodes_0)
        else:
            df = pd.DataFrame(m_pl_icd, index=nodes_1, columns=nodes_1)
            
        df.to_csv('path_length_nodes_'+type_proj+'.csv', index=True, header=True, sep=',', encoding = 'utf-8-sig')
        
        print("Average Shortest Path Length %s" % str(sum(pathlengths) / len(pathlengths)))
        print("\n")
        print('Calculating max value diameter ...')
        print("\n")
        print("Diameter: %s" % str(np.amax(m_pl_icd)))
        print("\n")
        print('Calculating assortativity ...')
        print("Assortativity: %s" % str(nx.degree_assortativity_coefficient(C)))
        print("\n")
        print('Calculating Betweenness Centrality')
        print("\n")
        b = nx.betweenness_centrality(C)
        sb = 0;
        for i in range(len(b)):
            sb = sb + list(b.items())[i][1]
        print("Betweenness: %s" % str(sb/len(b)))