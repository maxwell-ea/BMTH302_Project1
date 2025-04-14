"""
© 2025 Emily Maxwell Outland <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

three_crossover_trial.py

Last Modified: 04/14/2025
"""
import stat

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

    u_urdfs = starting_body_urdfs
    s_urdfs = starting_body_urdfs
    d_urdfs = starting_body_urdfs

    # Find the fitness for each body (final distance from starting point)
    fitness = [None] * num_bodies
    for i in range(len(starting_body_urdfs)):
        distance = sb.simulate_body(starting_body_urdfs[i])
        fitness[i] = distance[-1]

    u_fitness = fitness
    s_fitness = fitness
    d_fitness = fitness

    # Create three genetic algorithms, each with a different method of crossover
    uniform = Recombination(starting_bodies, u_fitness, prob_reproduction, prob_mutation, mutation_deviation,
                            encoding_type, minimise, crossover_method=0)

    single = Recombination(starting_bodies, s_fitness, prob_reproduction, prob_mutation, mutation_deviation,
                            encoding_type, minimise, crossover_method=1)

    double = Recombination(starting_bodies, d_fitness, prob_reproduction, prob_mutation, mutation_deviation,
                            encoding_type, minimise, crossover_method=2)

    # Instantiate the most fit for each genetic algorithm
    error = (None, None)
    u_most_fit = uniform.getMostFit()
    s_most_fit = single.getMostFit()
    d_most_fit = double.getMostFit()

    while u_most_fit == error:
        most_fit = uniform.getMostFit()

    while s_most_fit == error:
        most_fit = single.getMostFit()

    while d_most_fit == error:
        most_fit = double.getMostFit()

    # Create a pandas dataframe for each method to log info related to fitness
    columns = ('Generation', 'Fitness', 'body_w', 'body_l', 'body_h', 'leg_w1', 'leg_w2', 'leg_w3', 'leg_w4', 'leg_l1',
               'leg_l2', 'leg_l3', 'legl_4', 'legh_1', 'legh_2', 'legh_3', 'legh_4')

    u_data = [0, u_most_fit[1]]
    u_data.extend(u_most_fit[0])

    u_df = pd.DataFrame(columns=columns)
    u_df.loc[0] = u_data

    s_data = [0, u_most_fit[1]]
    s_data.extend(u_most_fit[0])

    s_df = pd.DataFrame(columns=columns)
    s_df.loc[0] = u_data

    d_data = [0, u_most_fit[1]]
    d_data.extend(u_most_fit[0])

    d_df = pd.DataFrame(columns=columns)
    d_df.loc[0] = u_data

    # Generational loop for genetic algorithm
    for i in range(generations):
        u_output = uniform.cycle()
        s_output = single.cycle()
        d_output = double.cycle()

        outputs = [u_output, s_output, d_output]

        print(f"Generation {i+1} of {generations}")

        for j in range(len(outputs)):

            output = outputs[j]

            bodies = output[0]
            individual = output[1]

            if j == 0:
                u_urdfs[individual] = evo.generate_urdf(bodies[individual], individual)
                u_fitness[individual] = sb.simulate_body(u_urdfs[individual])[-1]

                uniform.setFitness(u_fitness)

                # Add most fit member of the population to dataframe
                most_fit = uniform.getMostFit()
                new_data = [i + 1, most_fit[1]]
                new_data.extend(most_fit[0])
                u_df.loc[len(u_df.index)] = new_data

            elif j == 1:
                s_urdfs[individual] = evo.generate_urdf(bodies[individual], individual)
                s_fitness[individual] = sb.simulate_body(s_urdfs[individual])[-1]

                single.setFitness(s_fitness)

                # Add most fit member of the population to dataframe
                most_fit = single.getMostFit()
                new_data = [i + 1, most_fit[1]]
                new_data.extend(most_fit[0])
                s_df.loc[len(s_df.index)] = new_data

            else:
                d_urdfs[individual] = evo.generate_urdf(bodies[individual], individual)
                d_fitness[individual] = sb.simulate_body(d_urdfs[individual])[-1]

                double.setFitness(d_fitness)

                # Add most fit member of the population to dataframe
                most_fit = double.getMostFit()
                new_data = [i + 1, most_fit[1]]
                new_data.extend(most_fit[0])
                d_df.loc[len(d_df.index)] = new_data


    # Set the generation as the index in the dataframe
    u_df.set_index('Generation', inplace=True)
    print(u_df)

    # Print best fitness and most fit body
    u_best_body = uniform.getMostFit()
    print(f"Uniform Best Fitness: {u_best_body[1]}, Body: {u_best_body[0]}")

    u_df.to_csv(f'uniform_trial_{title}.csv')

    # Repeat process for single-point crossover
    s_df.set_index('Generation', inplace=True)
    print(s_df)

    # Print best fitness and most fit body
    s_best_body = single.getMostFit()
    print(f"Single Best Fitness: {s_best_body[1]}, Body: {s_best_body[0]}")

    s_df.to_csv(f'single_trial_{title}.csv')

    # Repeat process for two-point crossover
    d_df.set_index('Generation', inplace=True)
    print(d_df)

    # Print best fitness and most fit body
    d_best_body = double.getMostFit()
    print(f"Double Best Fitness: {d_best_body[1]}, Body: {d_best_body[0]}")

    d_df.to_csv(f'single_trial_{title}.csv')

    return [[uniform.population, uniform.fitness], [single.population, single.fitness],
            [double.population, double.fitness]]