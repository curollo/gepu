#!/usr/bin/env python3
import numpy as np
import copy
from gepu import Population, Terminal

def roulette_wheel(population):
    # calculate total and cumulative fitness of the population
    total = np.sum(individual.fitness for individual in population.individuals)
    cumulative = np.cumsum([individual.fitness / total for individual in population.individuals])

    # spin the wheel
    selected_individuals = []
    for _ in range(population.n):
        r = np.random.rand()
        selected_individual = next(individual for individual, cf in zip(population.individuals, cumulative) if r <= cf)
        selected_individuals.append(copy.deepcopy(selected_individual))

    selected_population = Population()
    selected_population.individuals = selected_individuals
    selected_population.n = population.n
    selected_population.pset = population.pset
    selected_population.head_length = population.head_length
    return(selected_population)

def mutate(population, mutations_per_individual=2):
    """ basic mutation of primitives """
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

def invert(population):
    """ invert part of genome """
    head_length = population.head_length
    if head_length > 2:
        selected_individuals = np.random.choice(population.individuals, size=int(.1 * population.n), replace=False)
        for individual in selected_individuals:
            start = np.random.randint(0, head_length - 2)
            stop = np.random.randint(start+1, head_length)
            individual.gene.genome[start: stop+1] = reversed(individual.gene.genome[start: stop+1])

def get_next_generation(population):
    # user evaluates fitness
    selected_population = roulette_wheel(population)
    mutate(selected_population)
    invert(selected_population)
    return selected_population
