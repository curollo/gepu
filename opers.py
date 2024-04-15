#!/usr/bin/env python3
import numpy as np
import copy
from gepu import Population, Terminal

def random_select(population):
    # random selection of individuals
    selected_individuals = np.random.choice(population.individuals, population.n, replace=False)
    selected_individuals = [copy.deepcopy(individual) for individual in selected_individuals]

    selected_population = Population()
    selected_population.individuals = selected_individuals
    selected_population.n = population.n
    selected_population.pset = population.pset
    selected_population.head_length = population.head_length
    return(selected_population)

def mutate(population, mutations_per_individual=2):
    # basic mutation of primitives
    pset = population.pset
    head_length = population.head_length
    arity = 2
    tail_length = head_length * (arity - 1) + 1
    for individual in population.individuals:
        for _ in range(mutations_per_individual):
            mutation_index = np.random.randint(0, head_length + tail_length - 2)
            if mutation_index < head_length:
                # mutate function or terminals in head
                if np.random.random() < 0.5:
                    individual.gene.genome[mutation_index] = np.random.choice(pset.functions)
                else:
                    individual.gene.genome[mutation_index] = Terminal(np.random.random())
            else:
                # mutate terminals in the tail
                individual.gene.genome[mutation_index] = Terminal(np.random.random())

def get_next_generation(population):
    # user evaluates fitness
    selected_population = random_select(population)
    mutate(selected_population)
    return selected_population
