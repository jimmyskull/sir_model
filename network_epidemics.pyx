# -*- coding: utf-8 -*-
import logging

import pandas as pd
import numpy as np

import igraph


class SIRState(object):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2


class SIRModel(object):

    def __init__(self, network, infection_rate=0.08, recover_rate=0.01,
                 ini_infected=1, ini_recovered=1, random_state=None):
        self.network = network
        self.nsize_ = network.vcount()
        self.infection_rate = infection_rate
        self.recover_rate = recover_rate
        self.ini_infected = ini_infected
        self.ini_recovered = ini_recovered
        self.random_state = random_state

    def states_as_pandas(self):
        colnames = ['susceptible', 'infected', 'recovered']
        pd_list = [pd.DataFrame(x, index=colnames, columns=[i]).transpose()
                   for i, x in enumerate(self.states_)]
        return pd.concat(pd_list, axis=0)

    def _update_state(self, state):
        self.state_ = state
        self.states_.append(np.bincount(self.state_, minlength=3))

    def init(self):
        np.random.seed(self.random_state)
        # list of vertices
        vertices = np.arange(0, self.nsize_)
        # select which one are intially recovered
        r_ini_count = min(self.ini_recovered, self.nsize_)
        r_idx = np.random.choice(vertices, r_ini_count, replace=False)
        # select which one are initially recovered
        s_vertices = np.delete(vertices, r_idx)
        s_ini_count = min(self.ini_infected, s_vertices.shape[0])
        i_idx = np.random.choice(s_vertices, s_ini_count, replace=False)
        # Build the initial state
        state = np.full(self.nsize_, SIRState.SUSCEPTIBLE)
        state[i_idx] = SIRState.INFECTED
        state[r_idx] = SIRState.RECOVERED
        self.states_ = list()
        self._update_state(state)

    def simulate_epoch(self):
        next_state = np.copy(self.state_)
        inf_idx = np.where(self.state_ == SIRState.INFECTED)[0]
        rec_idx = np.where(self.state_ == SIRState.RECOVERED)[0]

        if inf_idx.shape[0] == 0:
            logging.warning('No infected s_vertices. Returning the same state')
            return self.state_

        # Infect susceptible vertices
        infected_count = 0
        for vid in inf_idx:
            # Find which neighbors of |vid| are susceptible
            nei = np.array(self.network.neighbors(vid, mode=igraph.OUT))
            nei_idx = np.where(self.state_[nei] == SIRState.SUSCEPTIBLE)[0]
            s_count = nei_idx.shape[0]
            # Go to the next vertex if there are no susceptible vertices
            if s_count == 0:
                continue
            infected = nei_idx[np.random.rand(s_count) <= self.infection_rate]
            newly_infected_count = infected.shape[0]
            if newly_infected_count > 0:
                infected_count += newly_infected_count
                indices = nei[infected]
                next_state[indices] = SIRState.INFECTED

        logging.info(f'New infections: {infected_count}')

        # Recover infected ones
        i_count = inf_idx.shape[0]
        recovered = inf_idx[np.random.rand(i_count) <= self.recover_rate]
        if recovered.shape[0] > 0:
            next_state[recovered] = SIRState.RECOVERED

        self._update_state(next_state)

    def simulate(self, epochs=1):
        self.init()
        logging.info(f'Initial state: {np.bincount(self.state_)}')
        for epoch in range(epochs):
            self.simulate_epoch()
            logging.info(f'Epoch #{epoch + 1}: {np.bincount(self.state_)}')


