# -*- coding: utf-8 -*-
import logging

import numpy as np


class NetworkSIR(object):

    def __init__(self, network, initially_infected=1, initially_recovered=1,
                 random_state=None):
        self.network = network
        self.initially_infected = initially_infected
        self.initially_recovered = initially_recovered

    def init(self):

    def simulate_epoch(self, epochs=1):
