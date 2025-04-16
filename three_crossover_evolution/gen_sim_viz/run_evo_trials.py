"""
© 2024 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

run_evo_trials.py

Last Modified: 11/20/2024
"""

import evolution_trial as evo

num_bodies = 20
generations = 1000
trials = 1

for i in range(trials):
    actual_trial = i + 1
    print("Starting trial {}".format(actual_trial))

    final_state = evo.body_trial(num_bodies, generations, f"{actual_trial}")

    print("Finished trial {}".format(actual_trial))

print("Evolve Body Trials Complete")
