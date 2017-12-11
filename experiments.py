import random
import logging

import numpy as np

from sklearn.externals.joblib import Parallel, delayed

from igraph import Graph

import network_epidemics


def simulate_sir(epochs, random_state):
    random.seed(random_state)  # for igraph

    POPULATION = 4600000
    NEIGHBORHOOD = 5
    network = Graph.Watts_Strogatz(1, POPULATION, NEIGHBORHOOD, 0.02)

    sir = network_epidemics.SIRModel(
        network,
        infection_rate=0.15,
        recover_rate=0.02,
        ini_infected=500,
        ini_recovered=0,
        random_state=random_state)

    sir.simulate(epochs)
    return sir.states_as_pandas()


def experiment_network_sir(epochs=12, random_states=None, n_jobs=-1):
    if isinstance(random_states, list):
        random_states = np.array(random_states)

    parallel = Parallel(n_jobs=n_jobs)

    results = parallel(
        delayed(simulate_sir)(epochs, seed) for seed in random_states)

    return results