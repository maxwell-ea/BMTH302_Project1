"""
© 2024 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

Last Modified: 7/24/2024

Genetic Algorithms Developed for AFRL Research 2024

This package, containing the GeneticAlgorithm parent class and its children, was created to make genetic algorithm models
that are completely agnostic to the fitness function and to the details of the population, individual genetics, or
encoding.

Each model takes, in addition to parameters needed to initialize each model, only:
    * a list of fitness values
    * and a list of individuals where each individual is a list of binary values (also known as the population)

The fitness function (and therefore calculating the fitness of the population) and the encoding/decoding of genetics
is to be handled outside the GeneticAlgorithm class and its children. This is to account for complex simulation-based
fitness calculations, which could be calculated in other programs/programming languages, and so all models and their
methods will be able to handle any properly encoded population.

Distribution Statement: Distribution C / CUI
"""

from genalgs.GeneticAlgorithm import GeneticAlgorithm
from genalgs.Microbial import Microbial
