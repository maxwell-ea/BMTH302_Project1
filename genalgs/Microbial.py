"""
© 2024 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

Microbial.py

Last Modified: 7/24/2024

Distribution Statement: Distribution A
"""
import random

import numpy as np
import random as rd
from genalgs import GeneticAlgorithm


class Microbial(GeneticAlgorithm):
    """ A Microbial Genetic Algorithm using Localized Tournament Selection, child class of GeneticAlgorithm.

        Algorithm Methodology:
            1. Choose at random two individuals from the same deme (local "neighborhood")
            2. Compare the fitness of the individuals, and assign one winner and one loser
            3. Infect the loser with some proportion of the winner\'s genes
            4. With some probability, mutate the loser through a random bit flip or small percentage deviation, based
            on encoding type

        For inherited attributes and methods, refer to the GeneticAlgorithm class documentation.

        Attributes
        ----------
        mutation_deviation : float
            The percent to deviate a gene by when mutation is performed
        encoding_type : int
            If the genotype uses binary encoding, the encoding type is 0. If the genotype uses real-value encoding, the
            encoding type is 1.
        deme_size : int
            The size of the local neighborhood (or deme) from which the second parent is chosen

        Methods
        -------
        select(verbosity=0)
            Chooses at random two individuals from the same deme (local "neighborhood")
        reproduce(index1, index2, verbosity=0)
            Takes indices of two individuals in the population, assigns a winner and loser based on fitness, and infects
            each of the loser\'s genes with the winner\'s with probability self.prob_reproduction
        mutate(index, verbosity=0)
            With some probability self.prob_mutation, a random bit will be flipped for the individual in the
            population at the given index
        cycle(verbosity=0)
            Completes one generational cycle of selection, reproduction, and mutation then returns the new population
            and the list of what member of the population was infected/replaced
    """

    def __init__(self, initial_population, fitness, prob_reproduction, prob_mutation, mutation_deviation=0.01,
                 encoding_type=0, minimise=False, name="Microbial", deme_size: int = None):
        """ Calls __init__ from the parent class (GeneticAlgorithm) to set all attributes inherited from the parent.
        (For inherited attributes, refer to the GeneticAlgorithm class documentation.) Then, it sets self.deme_size,
        which is unique to the Microbial Genetic Algorithm.

        Parameters
        ----------
        self.deme_size : int
            The deme size will be defined as the number of individuals *on each side* of the first parent chosen
            in selection included in the deme (local "neighborhood") for that parent. (Ex. for a population size of
            11, a deme of 5 would include the entire population.) If the deme is not passed in, or it would include
            the entire population, it is set to zero

        self.mutation_deviation : float
            Only used for a real-valued encoding. This is the percentage of the current value that is added or subtracted
            during a mutation. [Ex. if mutation_deviation = 0.01, then when that gene is mutated it becomes:
            gene += 0.01 * gene (* +- 1)]

        self.encoding_type : int
            If encoding_type = 0, then the encoding is in binary, so mutation is a bit flip
            If encoding_type = 1, then the encoding is a real-valued number, so mutation is +- some percentage of the
            current value
        """

        super().__init__(initial_population, fitness, prob_mutation, prob_reproduction, minimise, name)

        if deme_size is None:
            self.deme_size = 0
        elif deme_size >= np.floor(self.population_size / 2):
            self.deme_size = 0
        else:
            self.deme_size = deme_size

        self.mutation_deviation = mutation_deviation
        self.encoding_type = encoding_type
        self.loser_index = -1

    def __str__(self):
        """ Custom method for string representation of the Microbial Genetic Algorithm

            Returns a string with info from the parent class, GeneticAlgorithm, and the deme size in addition.
        """
        if self.deme_size:
            deme = f''', and deme size {self.deme_size}'''
        else:
            deme = ', and deme size includes whole population'
        return super().__str__() + deme

    def select(self, verbosity=0) -> tuple[int, int]:
        """Randomly selects two individuals from the same deme (local "neighborhood").

            The first parent is selected randomly from the entire population, and then, the second parent is selected
            randomly from a range of individuals around the first parent of size self.deme_size, which mimics how
            biological systems tend to reproduce with individuals in their "local neighborhood", or deme.

            Parameters
            ----------
            verbosity : int, optional
                An int of values 0, 1, or 2 corresponding to the verbosity of the printout. When used as intended,
                verbosity will be passed in to the "cycle()" method, and that verbosity will be passed on to this
                method also. There will only be a printout if the verbosity = 2.

            Returns
            -------
            tuple[int, int]
                The indices of the two individuals selected from the population
        """

        index = rd.randrange(0, self.population_size)
        index2 = index

        while index == index2:
            if self.deme_size:
                deme_offset = rd.randint(-self.deme_size, self.deme_size)
                index2 = (index + deme_offset) % self.population_size
            else:
                index2 = rd.randrange(0, self.population_size)

        if verbosity == 2:
            print(f"Population Members Selected: {self.population[index]} (index = {index}, "
                  f"score = {self.fitness[index]}), {self.population[index2]}, (index = {index2}, "
                  f"score = {self.fitness[index2]})")

        return index, index2

    def reproduce(self, index1, index2, verbosity=0) -> int:
        """ With some probability self.prob_reproduction, each gene of the more fit individual will replace the gene of
            the less fit individual.

            First, the two individuals are compared, and the one with the higher fitness is declared the winner. Then,
            for each gene, a value will be chosen from a uniform distribution and compared to self.prob_reproduction.
            If it is <= self.prob_reproduction, the winner\'s gene will replace the loser\'s gene.

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

        for i in range(len(winner)):
            infect = np.random.uniform(0, 1)

            if infect <= self.prob_reproduction:
                loser[i] = winner[i]

        if verbosity == 2:
            print(f'''Infected Loser: {loser} (replaces individual at index {replace})''')

        self.population[replace] = loser
        self.replaced[replace] = 1
        self.loser_index = replace

        return self.loser_index

    def bit_mutate(self, index, verbosity: int = 0):
        """ With some probability prob_mutation, a random gene/bit will be flipped in the individual at the given
            index in the population.

            Parameters
            ----------
            index : int
                The index of the individual in the population to be mutated
            verbosity : int, optional
                An int of values 0, 1, or 2 corresponding to the verbosity of the printout. When used as intended,
                verbosity will be passed in to the "cycle()" method, and that verbosity will be passed on to this
                method also. There will only be a printout if the verbosity = 2.
        """

        loser = self.population[index]
        mutate = np.random.uniform(0, 1)

        if mutate <= self.prob_mutation:
            to_flip = rd.randrange(0, len(loser))

            if loser[to_flip]:
                loser[to_flip] = 0
                if verbosity == 2:
                    print(f"Mutation from 1 -> 0 at gene {to_flip}")
                    print(f"Infected, mutated individual: {loser} (index = {index})")
            else:
                loser[to_flip] = 1
                if verbosity == 2:
                    print(f'''Mutation from 0 -> 1 at gene {to_flip}''')
                    print(f'''Infected, mutated individual: {loser} (index = {index})''')

    def real_mutate(self, index, verbosity: int = 0):
        loser = self.population[index]
        mutate = np.random.uniform(0, 1)

        if mutate <= self.prob_mutation:
            to_mutate = rd.randrange(0, len(loser))

            loser[to_mutate] += loser[to_mutate] * self.mutation_deviation * random.choice([-1, 1])

            if verbosity == 2:
                print(f"Mutation at gene {to_mutate}")
                print(f"Infected, mutated individual: {loser} (index = {index})")

    def cycle(self, verbosity: int = 0):
        """ Completes one generational cycle of selection, reproduction, and mutation then returns the new population
            and the list of what member of the population was infected/replaced.

            Parameters
            ----------
            verbosity : int, optional
                An int of values 0, 1, or 2 corresponding to the verbosity of the printout:
                verbosity = 0 - no additional printouts other than "Evolution Complete" which will always be
                printed after one cycle;
                verbosity = 1 - prints out the new individual and its index;
                verbosity = 2 - one or more print statements for each stage of the generational cycle

            Returns
            -------
            tuple[list[list[int]], list[int]]
                The new population and a list of what member of the population was infected/replaced (where a 1
                indicates that the member of the population at the index was replaced and a 0 indicates no change)
        """

        # Reset replaced list at the start of each cycle
        self.replaced = [0] * self.population_size

        # Step 1: Choose 2 individuals through selection
        ind1, ind2 = self.select(verbosity)

        # Step 2: Infect the weaker individual with genes from the stronger individual
        loser_index = self.reproduce(ind1, ind2, verbosity)

        # Step 3: Mutate the infected individual
        if self.encoding_type:
            self.real_mutate(loser_index, verbosity)
        else:
            self.bit_mutate(loser_index, verbosity)

        if verbosity == 1:
            print(f"Evolved Individual at index {loser_index}: {self.population[loser_index]}\n")

        # print("Evolution Complete\n")

        # Return the new population and the list of which individuals (in this case only 1) were replaced
        return self.population, self.loser_index
