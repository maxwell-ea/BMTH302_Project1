"""
© 2025 Emily Maxwell Outland <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

visualize_trial.py

Last Modified: 04/14/2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import three_crossover_evolution.gen_sim_viz.evolution_trial as evo
import simulate_body_gui as sbg


def visualize_trial(trial_number: int, crossover_type: str):
    filename = f"gen_sim_viz/{crossover_type}_trial_{trial_number}.csv"
    df = pd.read_csv(filename)

    fitness = np.array(df.Fitness)

    plt.plot(fitness)
    plt.xlabel("Generation")
    plt.ylabel("Distance from Start (Fitness)")
    plt.title(f"Fitness over generations in {crossover_type} crossover, Trial {trial_number}")
    plt.show()


def load_simulate_body(trial_number: int, crossover_type: str):
    filename = f"gen_sim_viz/{crossover_type}_trial_{trial_number}.csv"
    df = pd.read_csv(filename)
    last = np.array(df.loc[len(df)-1]['body_w':'legh_4'])

    urdf = evo.generate_urdf(last, trial_number)
    distance = sbg.simulate_body_gui(urdf)

    return distance


def crossover_results(trial_number: int, crossover_type: str):
    visualize_trial(trial_number, crossover_type)
    load_simulate_body(trial_number, crossover_type)
