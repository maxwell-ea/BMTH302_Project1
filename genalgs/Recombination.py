"""
© 2025 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

Recombination.py

Last Modified: 04/09/2025
"""
import random

import numpy as np
import random as rd
from genalgs import Microbial


def point_crossover(winner, loser):
    """ A single point along the genotype will be randomly chosen, and from that point to the end of the gene, the
        genes of the more fit individual (winner) will replace the genes of the less fit individual (loser).

        Parameters
        ----------
        winner : list
            A list representing the winner's genotype
        loser : list
            A list representing the loser's genotype

        Returns
        -------
        list
            Returns the modified list representing the loser's genotype after gene crossover
    """

    point = rd.randint(0, len(winner) - 1)

    for i in range(point, len(winner)):
        loser[i] = winner[i]

    return loser


def two_point_crossover(winner, loser):
    """ A single point along the genotype will be randomly chosen, and from that point to the end of the gene, the
        genes of the more fit individual (winner) will replace the genes of the less fit individual (loser).

        Parameters
        ----------
        winner : list
            A list representing the winner's genotype
        loser : list
            A list representing the loser's genotype

        Returns
        -------
        list
            Returns the modified list representing the loser's genotype after gene crossover
    """

    points = sorted(rd.sample(range(0, len(winner)), 2))

    for i in range(points[0], points[1] + 1):
        loser[i] = winner[i]

    return loser


class Recombination(Microbial):
    """ A Genetic Algorithm using Localized Tournament Selection, variable methods of genetic crossover, child class
    of Microbial. This class was created to explore different kinds of recombination techniques.

        Algorithm Methodology:
            1. Choose at random two individuals from the same deme (local "neighborhood")
            2. Compare the fitness of the individuals, and assign one winner and one loser
            3. Three options for genetic crossover: (1) single-point crossover, (2) two-point crossover, (3) uniform
            crossover
            4. With some probability, mutate the loser through a random bit flip or small percentage deviation, based
            on encoding type

        For inherited attributes and methods, refer to the Microbial and GeneticAlgorithm class documentation.

        Attributes
        ----------
        crossover_method : int
            Chooses the strategy for genetic recombination

        Methods
        -------
        reproduce(index1, index2, verbosity=0)
            Takes indices of two individuals in the population, assigns a winner and loser based on fitness,
    """

    def __init__(self, initial_population, fitness, prob_reproduction=0.5, prob_mutation=0.05, mutation_deviation=0.01,
                 encoding_type=0, minimise=False, name="Recombination", deme_size: int = None, crossover_method=0):
        """ Calls __init__ from the parent class (Microbial) to set all attributes inherited from the parent.
        (For inherited attributes, refer to the Microbial and GeneticAlgorithm class documentation.) Then, it sets
        self.crossover_method, which is unique to the Recombination class.

        Parameters
        ----------
        self.crossover_method : int
            The strategy for genetic recombination where:
                * 0 is uniform crossover
                * 1 is single-point crossover
                * 2 is two-point crossover
        """

        super().__init__(initial_population, fitness, prob_reproduction, prob_mutation, mutation_deviation,
                         encoding_type, minimise, name, deme_size)

        self.crossover_method = crossover_method

    def uniform_crossover(self, winner, loser):
        """ With some probability self.prob_reproduction, each gene of the more fit individual (winner) will replace
            the gene of the less fit individual (loser).

            Parameters
            ----------
            winner : list
                A list representing the winner's genotype
            loser : list
                A list representing the loser's genotype

            Returns
            -------
            list
                returns the modified list representing the loser's genotype after gene crossover
        """

        for i in range(len(winner)):
            infect = np.random.uniform(0, 1)

            if infect <= self.prob_reproduction:
                loser[i] = winner[i]

        return loser

    def reproduce(self, index1, index2, verbosity=0) -> int:
        """ The individual with the higher fitness will be declared the winner, and for the selected method of
            recombination, a portion of the loser's genotype will be replaced with the winner's genotype.

            Parameters
            ----------
            index1 : int
                The index of the first individual in the population
            index2 : int
                The index of the second individual in the population
            verbosity : int, optional
                An int of values 0, 1, or 2 corresponding to the verbosity of the printout. When used as intended,
                verbosity will be passed in to the "cycle()" method, and that verbosity will be passed on to this
                method also. There will only be a printout if the verbosity = 2.

            Returns
            -------
            int
                Returns an index corresponding to which individual was infected (and therefore replaced) in the
                population by reproduction.
        """

        # Compare the fitnesses of the selected individuals and find the winner
        if self.minimise * self.fitness[index1] < self.minimise * self.fitness[index2]:
            winner = self.population[index1]
            loser = self.population[index2]

            replace = index2

            if verbosity == 2:
                print(f"Winner: {winner} (score = {self.fitness[index1]}), Loser: {loser} "
                      f"(score = {self.fitness[index2]})")
        else:
            winner = self.population[index2]
            loser = self.population[index1]

            replace = index1

            if verbosity == 2:
                print(f"Winner: {winner} (score = {self.fitness[index2]}), Loser: {loser} "
                      f"(score = {self.fitness[index1]})")

        # Based on the selected method of recombination, perform gene crossover
        if self.crossover_method == 0:
            loser = self.uniform_crossover(winner, loser)
        elif self.crossover_method == 1:
            loser = point_crossover(winner, loser)
        else:
            loser = two_point_crossover(winner, loser)

        if verbosity == 2:
            print(f'''Infected Loser: {loser} (replaces individual at index {replace})''')

        self.population[replace] = loser
        self.replaced[replace] = 1
        self.loser_index = replace

        return self.loser_index
