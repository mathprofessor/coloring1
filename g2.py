import random
import math
import pprint
import csv
import numpy as np
import os, sys
from ortools.sat.python import cp_model
import itertools as it
import csv
from pandas import DataFrame, read_csv
import pandas as pd 
import copy
import pickle
import time
import networkx as nx
import matplotlib.pyplot as plt


def embedIn(V,E,n,N):
    if n >= N:
        return(V,E)
    else:
        J = [j for j in range(N) if j >= n]
        V.extend(J)
        W = list(V)
        random.shuffle(W)
        sig = {}
        E1 = []
        for v in V:
            sig[v]=W[v]
        for (x,y) in E:
            (z,w) = (sig[x],sig[y])
            E1.append((z,w))
        return(V,E1)            
                


def xGraph(N,E):
    G = nx.Graph()
    for n in N:
        G.add_node(n)
    for (i,j) in E:
        G.add_edge(i,j)
    return G


    



def chromNum(N,E):
    n = len(N)
    model = cp_model.CpModel()
    x = {j : model.NewIntVar(0, n-1, 'x%i' % j) for j in N}
    z = model.NewIntVar(0, n-1, 'z' )
    
    for (i,j) in E:
        model.Add(x[i] != x[j])
    
    G = xGraph(N,E)
    cliqueList = list(nx.find_cliques(G))
    if len(cliqueList) == 0:
        exit('zero clique')
    cl = nx.graph_clique_number(G)
    for G2 in cliqueList:
        if len(G2) == cl:
            G1 = G2
            break
#   print('Clique size ',len(G2))
    col = 0
    for k in G1:
        model.Add(x[k] == col)
        col += 1
    print(col)
    
    for j in N:
        model.Add(x[j] <= z)
    model.Minimize(z+1)
    
    model.Add( z >= cl-1)
    
    
    
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers =14
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.SolveWithSolutionCallback(model, solution_printer)
    
    if status == cp_model.OPTIMAL:
        return (solver.ObjectiveValue(),cl*1.0)
    else:
        exit('Non Optimal Found')

def createGraph(n):
    nodes = [i for i in range(n)]
    edges = [(i,j) for i in nodes for j in nodes if i < j]
    random.shuffle(edges)
    
    for i in range(random.randint(0,len(edges))):
        edges.pop(0)
    
    
    return (nodes, edges)
    

def seeX(n,N):
    (V,E) = createGraph(n)
    A = np.zeros((N,N))
    for (i,j) in E:
        A[i,j] = 1
        A[j,i] = 1
#   print(A)
#   print(V)
#   print(E)
    A = A.ravel()
    B = list(A)
    G = xGraph(V,E) 

    out = chromNum(V,E)
    d = nx.coloring.greedy_color(G, strategy="largest_first")
    print(d)
    print("greedy color ", max(list(d.values()))+1)
    if out == None:
        print("error",V,E)
        exit()
    else:
        (ch,cl) = out
    rec = [ch,cl] + B
    print(ch,cl)
    nx.draw_circular(G)
#   print(rec)


def makeRec(filename,k,N):

    with open(filename, mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for n in range(N//2,N):
            for p in range(k):
                (V,E) = createGraph(n)
                out = chromNum(V,E)
                (V,E) = embedIn(V,E,n,N)
                A = np.zeros((N,N))
                
                for (i,j) in E:
                    A[i,j] = 1
                    A[j,i] = 1
                    
                A = A.ravel()
                B = list(A)

                
                if out == None:
                    print("error",V,E)
                    exit()
                else:
                    (ch,cl) = out
                rec = [ch,cl] + B
                csvwriter.writerow(rec)


#   nx.draw_circular(G)


def main():
#	seeX(25,30)
	
	makeRec("dev.csv",1000,50)
	makeRec("test.csv",1000,50)
	makeRec("train.csv",10000,50)
	
		
	
main()
