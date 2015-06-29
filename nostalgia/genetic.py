# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np

def calculate_fitness(g,t):
    return np.exp(-np.sqrt((g-t).dot(g-t)))

if __name__ == '__main__':
    population_size = 100    
    gen_size = 100
    target_gen = np.random.randint(0,2,gen_size)
    print target_gen
    gens = []
    for i in range(population_size):
        gens.append(np.random.randint(0,2,gen_size))
    
    fitness = []
    for i in range(population_size):
        fitness.append(calculate_fitness(gens[i],target_gen))
    print np.sum(fitness)
    
    for x in range(1000):
        fitness = []
        for i in range(population_size):
            fitness.append(calculate_fitness(gens[i],target_gen))
        normalize_fitness = fitness / np.sum(fitness)
        
        new_gens = []
        for i in range(population_size):
            parents = []
            for ii in range(2):   
                c = 0                
                s = np.random.uniform()
                for iii in range(population_size):
                    c += normalize_fitness[iii]
                    if s < c:
                        parents.append(gens[iii])
                        break
            child = []
            for ii in range(gen_size):
                if np.random.uniform() < 0.01:
                    child.append(np.random.randint(0,2))
                else:
                    if ii % 2 == 0:
                        child.append(parents[0][ii])
                    else:
                        child.append(parents[1][ii])
            new_gens.append(np.array(child))
        
        gens = new_gens
    
    fitness = []
    for i in range(population_size):
        fitness.append(calculate_fitness(gens[i],target_gen))
    print gens[10]
    print np.sum(fitness)
        
    