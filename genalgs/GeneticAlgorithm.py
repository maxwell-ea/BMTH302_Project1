"""
© 2024 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

GeneticAlgorithm.py

Last Modified: 7/25/2024

Distribution Statement: Distribution A
"""


class GeneticAlgorithm:
    """ A parent class for all Genetic Algorithms.

        The purpose of this parent class is to provide a common framework for attributes shared by all genetic
        algorithms and a few methods useful for setting, finding, and retrieving the fitness of the population.

        Although not strictly enforced, each child class will be responsible for implementing the following methods:
            * select() - selects individual(s) to reproduce from the population
            * reproduce() - uses the individual(s) to create a new offspring in some manner
            * mutate() - causes a mutation to the newly created offspring in some manner
            * cycle() - one generational cycle that includes selection, reproduction, mutation, and altering the old
            population to accommodate for the new individual(s)

        To note:
            1) Only cycle is intended for public use, but the implementation of the other methods is to maintain a
            common framework across all genetic algorithms for easily-understandable code.
            2) Other methods may be implemented, depending on the nature of the algorithm.

        Attributes
        ----------
        population : list[list[int]]
            A list of individuals where each individual is a list of binary values
        fitness : list[float]
            A list of fitness values, where a given index represents the fitness value of the individual at that
            same index in the population
        minimise : bool, optional
            Whether the genetic algorithm will minimise or maximise the fitness of the population. Default value is
            False.
        prob_mutation: float
            A float [0.0, 1.0) corresponding to the probability that the individual will be mutated
        prob_reproduction: float
            A float [0.0, 1.0) corresponding to the probability for reproduction or the proportion of gene
            exchange/crossover between two individuals (algorithm dependent)
        name : str, optional
            The name of the genetic algorithm (used for printing/ID purposes when there are multiple instantiations
            of the same model). Default value is "Parent".
        population_size : int
            The number of individuals in the population
        replaced : list[int]
            A list representing which individuals have been replaced as part of the reproduction process. There will
            be a 1 at an index if the individual at the corresponding index in the population has been replaced and
            a 0 if it has not been replaced.
        most_fit : list[int]
            The most fit individual in the population
        best_fitness : float
            The fitness of the most fit member of the population

        Methods
        -------
        setFitness(fitness: list[float])
            Sets self.fitness to the input fitness values

        getMostFit()
            Returns the most fit individual in the population and its fitness

        findMostFit()
            Determines the most fit individual in the population and its fitness and sets self.most_fit and
            self.best_fitness accordingly
    """

    def __init__(self, initial_population, fitness, prob_mutation, prob_reproduction, minimise=False, name="Parent"):
        """ Parameters
            ----------
            initial_population : list[list[int]]
                A list of individuals where each individual is a list of binary values, to be used as the starting
                population for the algorithm
            fitness : list[float]
                A list of fitness values, where a given index represents the fitness value of the individual at that
                same index in the population
            prob_mutation : float
                A float [0.0, 1.0) corresponding to the probability that the individual will be mutated
            prob_reproduction : float
                A float [0.0, 1.0) corresponding to the probability for reproduction or the proportion of gene
            minimise : bool, optional
                Whether the genetic algorithm will minimise or maximise the fitness of the population (default is False)
            name : str, optional
                The name of the genetic algorithm (default is 'Parent')
        """

        self.population = initial_population
        self.fitness = fitness
        self.minimise = 1 if minimise else -1
        self.prob_mutation = prob_mutation
        self.prob_reproduction = prob_reproduction
        self.name = name

        self.population_size = len(initial_population)
        self.replaced = [0] * self.population_size

        self.most_fit = None
        self.best_fitness = None
        self.findMostFit()

    def __str__(self):
        """ Custom method for string representation of the GeneticAlgorithm

        Returns a string with the following info:
            * name
            * population size
            * rate/probability of reproduction
            * rate/probability of mutation
            * minimise or not
        """

        minim = True if self.minimise == 1 else False
        return (f"{self.name} Genetic Algorithm with population size: {self.population_size}, reproduction rate: "
                f"{self.prob_reproduction}, mutation rate: {self.prob_mutation}, minimise = {minim}")

    def setFitness(self, fitness):
        """ Sets self.fitness to the input fitness values and recalculates self.best_fitness and self.most_fit.

            Parameters
            ----------
            fitness : list[float]
                An input of fitness values, where a given index represents the fitness value of the individual at that
                same index in the population
        """

        self.fitness = fitness
        self.findMostFit()

    def getMostFit(self):
        """ Returns the most fit individual in the population and its fitness.

            Returns
            -------
            tuple(list[int], int)
                The most fit individual in the population and the fitness of the most fit member of the population
        """

        return self.most_fit, self.best_fitness

    def findMostFit(self):
        """ Determines the most fit individual in the population and its fitness and sets self.most_fit and
        self.best_fitness accordingly.
        """

        current_best = self.fitness[0] * self.minimise

        for i in range(1, len(self.fitness)):
            current_fitness = self.fitness[i] * self.minimise
            if current_fitness < current_best:
                current_best = current_fitness
                self.best_fitness = self.fitness[i]
                self.most_fit = self.population[i]
