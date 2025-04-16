"""
© 2024 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

three_crossover_results.py

Last Modified: 11/21/2024
"""

from gen_sim_viz import visualize_trial as vt

trial_number = 5
crossover_type = 1

crossover_names = ["uniform", "single", "double"]

vt.crossover_results(trial_number, crossover_names[crossover_type])
