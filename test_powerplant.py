# %%
#!cat /etc/os-release
#!nvidia-smi
#!git clone https://github.com/curollo/gepu
#!pip install ucimlrepo
#%load_ext autoreload
#%autoreload 2
import numpy as np
import pandas as pd
import torch
import time

import gepu
from opers import *
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import mean_squared_error, r2_score

# set device dynamically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# load UCI Combined Cycle Power Plant dataset
combined_cycle_power_plant = fetch_ucirepo(id=294)
X = combined_cycle_power_plant.data.features
y = combined_cycle_power_plant.data.targets

print(combined_cycle_power_plant.metadata)
print(combined_cycle_power_plant.variables)

PowerPlant = pd.concat([X, y], axis=1)
PowerPlant.describe()

# split for training and validation
msk = np.random.rand(len(PowerPlant)) < 0.8
train = PowerPlant[msk]
holdout = PowerPlant[~msk]

# convert to tensors
AT = torch.from_numpy(train.AT.values).float().to(device)
V = torch.from_numpy(train.V.values).float().to(device)
AP = torch.from_numpy(train.AP.values).float().to(device)
RH = torch.from_numpy(train.RH.values).float().to(device)

Y = torch.from_numpy(train.PE.values).float().to(device)

# %%
# define tensor opers
def add(x, y): return torch.add(x, y)
def sub(x, y): return torch.sub(x, y)
def mul(x, y): return torch.mul(x, y)

pset = gepu.PrimitiveSet()
pset.add_function(add)
pset.add_function(sub)
pset.add_function(mul)
pset.add_terminal('AT')
pset.add_terminal('V')
pset.add_terminal('AP')
pset.add_terminal('RH')

# %%
# define number of total runs of the algorithm and log total evolution time
torch_time=[]
for i in range(1):
    start=time.time()

    # init population
    population = gepu.Population()

    # set number of individuals in population and head length
    population.generate(120, pset, 7)

    # start glorious evolution
    best_individual = None
    best_fitness = float('inf')

    # define number of generations
    for i in range(50):

        # apply genetic operators
        population = get_next_generation(population)

        # eval translated phenotype to tensors and stack them
        individual_tensors = [torch.tensor(eval(str(ind))).float().to(device) if torch.tensor(eval(str(ind))).size() != torch.Size([]) else torch.ones(len(Y)).float().to(device) for i, ind in enumerate(population.individuals)]
        individual_tensor = torch.stack(individual_tensors)

        # calculate MSE for entire population
        fitness_values = torch.mean((Y - individual_tensor) ** 2, dim=1).cpu().numpy()

        # update best_individual found during evolution
        for j, evaluated in enumerate(population.individuals):
            evaluated.fitness = fitness_values[j]
            if evaluated.fitness < best_fitness:
                best_individual = evaluated
                best_fitness = evaluated.fitness
                print(f'MSE = ', best_fitness)

    end=time.time()
    print('time:',end-start)
    torch_time.append(end-start)

# %%
# print best individual and it's fitness
print(best_individual)
print(best_fitness)

# %%
# eval model on holdout data
def model_output(AT, V, AP, RH, model): return eval(model)
def add(x, y): return np.add(x, y)
def sub(x, y): return np.subtract(x, y)
def mul(x, y): return np.multiply(x, y)
predPE = model_output(holdout.AT, holdout.V, holdout.AP, holdout.RH, str(best_individual))
predPE.describe()
predPE.head()

# %%
# get eval metrics
print("Mean squared error: %.2f" % mean_squared_error(holdout.PE, predPE))
print("R2 score : %.2f" % r2_score(holdout.PE, predPE))

# %%
# visualize found solution
from matplotlib import pyplot
pyplot.rcParams['figure.figsize'] = [20, 5]
plotlen=200
pyplot.plot(predPE.head(plotlen))       # blue predictions
pyplot.plot(holdout.PE.head(plotlen-2)) # orange actual values
pyplot.show()

# %%
# error distribution (simple average estimates)
pyplot.rcParams['figure.figsize'] = [10, 5]
hfig = pyplot.figure()
ax = hfig.add_subplot(111)
numBins = 100
ax.hist(holdout.PE-predPE,numBins,color='green',alpha=0.8)
pyplot.show()

# %%
# error distribution of average estimates and found solution
hfig3 = pyplot.figure()
ax = hfig3.add_subplot(111)
numBins = 100
ax.hist(holdout.PE-holdout.PE.mean(),numBins,color='green',alpha=0.8)
ax.hist(holdout.PE-predPE,numBins,color='orange',alpha=0.8)
pyplot.show()
