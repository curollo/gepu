#!/usr/bin/env python3
import numpy as np
import copy
from gepu import Population, Terminal, Function

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
    arity = 2
    tail_length = population.head_length * (arity - 1) + 1
    for individual in population.individuals:
        for _ in range(mutations_per_individual):
            mutation_index = np.random.randint(0, population.head_length + tail_length - 2)
            if mutation_index < population.head_length:
                # mutate function or terminals in head
                if np.random.random() < 0.5:
                    individual.gene.genome[mutation_index] = np.random.choice(population.pset.functions)
                else:
                    individual.gene.genome[mutation_index] = Terminal(np.random.random())
            else:
                # mutate terminals in the tail
                individual.gene.genome[mutation_index] = Terminal(np.random.random())

def invert(population, p=.1):
    """ invert part of genome """ 
    if population.head_length > 2:
        selected_individuals = np.random.choice(population.individuals, size=int(p * population.n), replace=False)
        for individual in selected_individuals:
            start = np.random.randint(0, population.head_length - 2)
            stop = np.random.randint(start+1, population.head_length)
            individual.gene.genome[start: stop+1] = reversed(individual.gene.genome[start: stop+1])

def transpose_is(population, p=.1):
    """ non-root transposition """
    if population.head_length > 2:
        selected_individuals = np.random.choice(population.individuals, size=int(p * population.n), replace=False)
        for individual in selected_individuals:
            is_length = np.random.randint(0, population.head_length - 1)
            start = np.random.randint(0, len(individual.gene.genome) - is_length)
            stop = start + is_length
            iseq = individual.gene.genome[start:stop + 1]
            insertion_point = np.random.randint(1, population.head_length - is_length)
            individual.gene.genome[insertion_point:insertion_point + is_length + 1] = iseq

def transpose_ris(population, p=.1):
    """ root transposition """
    if population.head_length > 2:
        selected_individuals = np.random.choice(population.individuals, size=int(p * population.n), replace=False)
        for individual in selected_individuals:
            function_idx = [i for i, p in enumerate(individual.gene.genome) if isinstance(p, Function)]
            if not function_idx: continue
            start = np.random.choice(function_idx)
            is_length = np.random.randint(2, population.head_length - 1)
            individual.gene.genome[0:is_length] = individual.gene.genome[start:start+is_length]

def get_gene_len(population):
    head_length = population.head_length
    arity = 2
    tail_length = head_length * (arity - 1) + 1
    return head_length + tail_length

def onep_recombine(population, p=.1):
    """ one point recombination """
    gene_length = get_gene_len(population)
    parents = np.random.choice(population.individuals, size=int(p * population.n), replace=False)
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        recomb_point = np.random.randint(0, gene_length - 1)
        parent1.gene.genome[recomb_point:], parent2.gene.genome[recomb_point:] = (
            parent2.gene.genome[recomb_point:], parent1.gene.genome[recomb_point:]
        )

def twop_recombine(population, p=0.1):
    """ two point recombination """
    gene_length = get_gene_len(population)
    parents = np.random.choice(population.individuals, size=int(.1 * population.n), replace=False)
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        recomb_point = np.random.randint(0, gene_length - 1)
        recomb_end = np.random.randint(recomb_point, gene_length - 1)
        parent1.gene.genome[recomb_point:recomb_end+1], parent2.gene.genome[recomb_point:recomb_end+1] = (
            parent2.gene.genome[recomb_point:recomb_end+1], parent1.gene.genome[recomb_point:recomb_end+1]
        )

def get_next_generation(population):
    # user evaluates fitness
    selected_population = roulette_wheel(population)
    mutate(selected_population)
    invert(selected_population)
    transpose_is(selected_population)
    transpose_ris(selected_population)
    onep_recombine(selected_population)
    twop_recombine(selected_population)
    return selected_population
