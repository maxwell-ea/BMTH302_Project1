"""
© 2025 Emily Maxwell Outland <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

run_crossover_trials.py

Last Modified: 04/14/2025
"""

import three_crossover_trial as cross

num_bodies = 20
generations = 1000
trials = 1

for i in range(trials):
    actual_trial = i + 1
    print(f"Starting trial {actual_trial}")

    final_state = cross.three_crossover_trial(num_bodies, generations, f"{actual_trial}")

    print(f"Finished trial {actual_trial}")

print("Crossover Trials Complete")
