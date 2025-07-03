#!/usr/bin/env python3
"""Quick test to verify the new recommender passes Test Case 1"""

import numpy as np
from test import test_1, required_results
from simulation import Simulation

# Import the new recommender (save the hybrid version as Recommender.py)
from Recommender import Recommender

print("Testing new hybrid recommender on Test Case 1...")
print(f"Required score: {required_results[0]}")
print("-" * 50)

# Run 5 trials
rewards = []
for i in range(5):
    sim = Simulation(test_1['P'], test_1['item_prices'], 
                     test_1['budget'], test_1['n_weeks'])
    reward = sim.simulate()
    rewards.append(reward)
    print(f"Trial {i+1}: {reward}")

print("-" * 50)
print(f"Mean: {np.mean(rewards):.0f}")
print(f"Std: {np.std(rewards):.0f}")
print(f"Min: {np.min(rewards)}")
print(f"Max: {np.max(rewards)}")

if np.mean(rewards) >= required_results[0]:
    print("\n✅ PASSES on average!")
else:
    print(f"\n❌ Still short by {required_results[0] - np.mean(rewards):.0f} on average")
    
# Also test on the other cases to ensure we didn't break anything
print("\n\nQuick check on other test cases:")
from test import test_2, test_3

sim2 = Simulation(test_2['P'], test_2['item_prices'], test_2['budget'], test_2['n_weeks'])
reward2 = sim2.simulate()
print(f"Test 2: {reward2} (required: {required_results[1]}) {'✅' if reward2 >= required_results[1] else '❌'}")

sim3 = Simulation(test_3['P'], test_3['item_prices'], test_3['budget'], test_3['n_weeks'])
reward3 = sim3.simulate()
print(f"Test 3: {reward3} (required: {required_results[2]}) {'✅' if reward3 >= required_results[2] else '❌'}")