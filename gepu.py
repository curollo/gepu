#!/usr/bin/env python3
import numpy as np

class PrimitiveSet:
    def __init__(self):
        self.functions = []
        self.terminals = []

    def add_function(self, function):
        self.functions.append(Function(function))

    def add_terminal(self, variable):
        self.terminals.append(Terminal(variable))

    def choose_function(self):
        return np.random.choice(self.functions)

    def choose_terminal(self):
        if np.random.random() < 0.5 or not self.terminals:
            return Terminal(np.random.random())
        else:
            return np.random.choice(self.terminals)

class Function:
    def __init__(self, func):
        self.func = func
        self.arity = 2

    def format(self, *args):
        args = args[:self.arity]
        return (self.func.__name__ + '(' + ', '.join(map(str, args)) + ')')

    def __str__(self):
        return self.func.__name__

    def __repr__(self):
        return self.func.__name__

class Terminal:
    def __init__(self, value):
        self.value = value
        self.arity = 0

    def format(self):
        return str(self.value)

    def __str__(self):
        if isinstance(self.value, str):
            return self.value
        else:
            return str(round(self.value, 3))

    def __repr__(self):
        if isinstance(self.value, str):
            return self.value
        else:
            return str(round(self.value, 3))

def generate_genome(pset, head_length):
    functions = pset.functions
    arity = 2 #TODO: more than 2 maybe?
    tail_length = head_length * (arity - 1) + 1
    genome = [None] * (head_length + tail_length) # init genome

    # gen head part (functions and terminals)
    start = 0
    for i in range(start,head_length):
        if np.random.random() < 0.5:
            genome[i] = pset.choose_function()
        else:
            genome[i] = pset.choose_terminal()
    # gen tail part (only terminals)
    for i in range(head_length, head_length + tail_length):
        genome[i] = pset.choose_terminal()
    return genome

class Gene:
    def __init__(self, pset, head_length):
        self._head_length = head_length
        self.genome = generate_genome(pset, head_length)

    def get_kexpression(self):
        genome = self.genome
        expr = [genome[0]]
        i = 0
        j = 1
        while i < len(expr):
            for _ in range(genome[i].arity):
                expr.append(genome[j])
                j += 1
            i += 1
        return expr

    def __str__(self):
        kexpr = self.get_kexpression()
        i = len(kexpr) - 1 # start by the last item in the tree
        while i >= 0:
            # if the item is a function
            if kexpr[i].arity > 0:
                args = []
                for _ in range(kexpr[i].arity):
                    element = kexpr.pop() # remove last element
                    if isinstance(element, str):
                        # append to the args of the current function item
                        args.append(element)
                    else:
                        # append terminal to the args of the current function item
                        args.append(element.format())
                # when all args for this function has been acquired, format
                kexpr[i] = kexpr[i].format(*reversed(args))
            i -= 1
        # return formatted root
        return kexpr[0] if isinstance(kexpr[0], str) else kexpr[0].format()

class Individual:
    def __init__(self, pset, head_length):
        self.gene = Gene(pset, head_length)

    def __str__(self):
        return str(self.gene)

    def __repr__(self):
        return str(self.gene)

class Population:
    def __init__(self):
        self.individuals = None
        self.n = None
        self.pset = None
        self.head_length = None

    def generate(self, n, pset, head_length):
        self.individuals = [Individual(pset, head_length) for _ in range(n)]
        self.n = n
        self.pset = pset
        self.head_length = head_length
