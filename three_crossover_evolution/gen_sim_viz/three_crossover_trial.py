"""
© 2025 Emily Maxwell Outland <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

three_crossover_trial.py

Last Modified: 04/14/2025
"""

import simulate_body_nogui as sb
from genalgs import Recombination
import evolution_trial as evo

import numpy as np
import pandas as pd


def three_crossover_trial(num_bodies: int, generations: int, title: str, prob_reproduction=0.5, prob_mutation=0.1,
               mutation_deviation=0.05, encoding_type=1, minimise=False):

    # Generate bodies (list of parameters)
    starting_bodies = evo.randomize_bodies(num_bodies)

    # Generate urdfs of bodies
    starting_body_urdfs = evo.generate_urdfs(starting_bodies)

    # Find the fitness for each body (final distance from starting point)
    starting_fitness = [None] * num_bodies
    for i in range(len(starting_body_urdfs)):
        distance = sb.simulate_body(starting_body_urdfs[i])
        starting_fitness[i] = distance[-1]

    # Create three genetic algorithms, each with a different method of crossover
    uniform = Recombination(starting_bodies, starting_fitness, prob_reproduction, prob_mutation, mutation_deviation,
                            encoding_type, minimise, name="uniform", crossover_method=0)

    single = Recombination(starting_bodies, starting_fitness, prob_reproduction, prob_mutation, mutation_deviation,
                            encoding_type, minimise, name="single", crossover_method=1)

    double = Recombination(starting_bodies, starting_fitness, prob_reproduction, prob_mutation, mutation_deviation,
                            encoding_type, minimise, name= "double", crossover_method=2)

    methods = [uniform, single, double]

    for method in methods:
        ga = method

        body_urdfs = starting_body_urdfs
        fitness = starting_fitness

        print(f"Trial of {ga.name} crossover method")

        # Instantiate the most fit for each genetic algorithm
        error = (None, None)
        most_fit = ga.getMostFit()

        while most_fit == error:
            most_fit = uniform.getMostFit()

        # Create a pandas dataframe for each method to log info related to fitness
        columns = ('Generation', 'Fitness', 'body_w', 'body_l', 'body_h', 'leg_w1', 'leg_w2', 'leg_w3', 'leg_w4', 'leg_l1',
                   'leg_l2', 'leg_l3', 'legl_4', 'legh_1', 'legh_2', 'legh_3', 'legh_4')

        data = [0, most_fit[1]]
        data.extend(most_fit[0])

        df = pd.DataFrame(columns=columns)
        df.loc[0] = data

        # Generational loop for genetic algorithm
        for i in range(generations):
            output = ga.cycle()
            print(f"Generation {i + 1} of {generations}")

            bodies = output[0]
            individual = output[1]

            body_urdfs[individual] = evo.generate_urdf(bodies[individual], individual)
            fitness[individual] = sb.simulate_body(body_urdfs[individual])[-1]

            ga.setFitness(fitness)

            # Add most fit member of the population to dataframe
            most_fit = ga.getMostFit()
            new_data = [i + 1, most_fit[1]]
            new_data.extend(most_fit[0])
            df.loc[len(df.index)] = new_data


        # Set the generation as the index in the dataframe
        df.set_index('Generation', inplace=True)
        print(df)

        # Print best fitness and most fit body
        best_body = ga.getMostFit()
        print(f"{ga.name} method best fitness: {best_body[1]}, body: {best_body[0]}")

        df.to_csv(f'{ga.name}_trial_{title}.csv')

    return [[uniform.population, uniform.fitness], [single.population, single.fitness],
            [double.population, double.fitness]]