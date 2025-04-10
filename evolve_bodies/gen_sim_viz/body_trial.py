"""
© 2024 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

body_trial.py

Last Modified: 11/18/2024
"""

import evolve_bodies.gen_sim_viz.generate_body as gb
import simulate_body_nogui as sb
from genalgs import Microbial

import numpy as np
import pandas as pd

# Limits for body parameters
body_w_lim = 5
body_l_lim = 5
body_h_lim = 5
leg_w_lim = 5
leg_l_lim = 5
leg_h_lim = 5


def randomize_bodies(num_bodies: int):
    all_bodies = [None] * num_bodies

    for i in range(num_bodies):
        body = [np.random.uniform(0, body_w_lim), np.random.uniform(0, body_l_lim),
                np.random.uniform(0, body_h_lim)]

        body.extend(np.random.uniform(0, leg_w_lim, 4))
        body.extend(np.random.uniform(0, leg_l_lim, 4))
        body.extend(np.random.uniform(0, leg_h_lim, 4))

        all_bodies[i] = body

    return all_bodies


def generate_urdfs(bodies: list[list]):
    all_urdfs = [None] * len(bodies)

    for i in range(len(bodies)):
        body_urdf = gb.generate_body(i, bodies[i])

        all_urdfs[i] = body_urdf

    return all_urdfs


def generate_urdf(body: list, index: int = 0):
    body_urdf = gb.generate_body(index, body)
    return body_urdf


def simulate_body(body_urdf: str):
    return sb.simulate_body(body_urdf)


def body_trial(num_bodies: int, generations: int, title: str, prob_reproduction=0.8, prob_mutation=0.1,
               mutation_deviation=0.05, encoding_type=1, minimise=False):

    # Generate bodies (list of parameters)
    bodies = randomize_bodies(num_bodies)

    # Generate urdfs of bodies
    body_urdfs = generate_urdfs(bodies)

    # Find the fitness for each body (final distance from starting point)
    fitness = [None] * num_bodies
    for i in range(len(body_urdfs)):
        distance = simulate_body(body_urdfs[i])
        fitness[i] = distance[-1]

    ga = Microbial(bodies, fitness, prob_reproduction, prob_mutation, mutation_deviation, encoding_type, minimise)

    error = (None, None)
    most_fit = ga.getMostFit()

    while most_fit == error:
        most_fit = ga.getMostFit()

    # Create pandas dataframe to info related to fitness
    columns = ('Generation', 'Fitness', 'body_w', 'body_l', 'body_h', 'leg_w1', 'leg_w2', 'leg_w3', 'leg_w4', 'leg_l1',
               'leg_l2', 'leg_l3', 'legl_4', 'legh_1', 'legh_2', 'legh_3', 'legh_4')

    data = [0, most_fit[1]]
    data.extend(most_fit[0])

    df = pd.DataFrame(columns=columns)
    df.loc[0] = data

    # Generational loop for genetic algorithm
    for i in range(generations):
        output = ga.cycle()
        print(f"Generation {i+1} of {generations}")

        bodies = output[0]
        individual = output[1]

        body_urdfs[individual] = generate_urdf(bodies[individual], individual)
        fitness[individual] = simulate_body(body_urdfs[individual])[-1]

        ga.setFitness(fitness)

        # Add most fit member of the population to dataframe
        most_fit = ga.getMostFit()
        new_data = [i+1, most_fit[1]]
        new_data.extend(most_fit[0])
        df.loc[len(df.index)] = new_data

    # Set the generation as the index in the datafram
    df.set_index('Generation', inplace=True)
    print(df)

    # Print best fitness and most fit body
    best_body = ga.getMostFit()
    print(f"Best Fitness: {best_body[1]}, Body: {best_body[0]}")

    df.to_csv(f'body_trial_{title}.csv')

    return bodies, fitness
