# simple_parameter_test.py
import numpy as np
from test import test_1, test_2, test_3
from simulation import Simulation

print("Simple Parameter Testing")
print("="*60)
print("\nThis script will help you find the best parameters.")
print("After each test, manually update your Recommender.py with the suggested values.")
print("="*60)

# Choose test case
test_num = input("\nWhich test case? (1/2/3, default=1): ") or "1"
test_cases = {'1': test_1, '2': test_2, '3': test_3}
test_case = test_cases[test_num]

print(f"\nTesting on test_{test_num}...")
print("Run simulation.py after each suggested change to see the effect.")

# Current baseline
print("\n1. BASELINE TEST")
print("Run: python simulation.py")
baseline = input("What was your baseline score? ")

# Test 1: Exploration period
print("\n2. TEST EXPLORATION PERIOD")
print("Try these values for exploration period:")
print("""
Modify your Recommender's recommend() method:
- Change: if self.current_round < 50:
- To:     if self.current_round < 20:

Run simulation.py and note the score.
Then try < 30, < 40, etc.
""")

best_exploration = input("Which exploration period gave the best score? ")

# Test 2: Initial priors
print("\n3. TEST INITIAL PRIORS")
print(f"Keep exploration period at {best_exploration}")
print("""
In __init__, try these combinations:
a) self.alpha = np.ones((n_users, self.n_items)) * 1.0
   self.beta = np.ones((n_users, self.n_items)) * 1.0

b) self.alpha = np.ones((n_users, self.n_items)) * 1.5
   self.beta = np.ones((n_users, self.n_items)) * 0.5

c) self.alpha = np.ones((n_users, self.n_items)) * 2.0
   self.beta = np.ones((n_users, self.n_items)) * 0.5

d) self.alpha = np.ones((n_users, self.n_items)) * 2.0
   self.beta = np.ones((n_users, self.n_items)) * 1.0
""")

best_prior = input("Which prior combination (a/b/c/d) gave the best score? ")

# Test 3: Empirical weighting
print("\n4. TEST EMPIRICAL WEIGHTING")
print("If you're using Thompson Sampling with empirical blending:")
print("""
Look for code like:
sampled_probs[i, j] = 0.7 * sampled_probs[i, j] + 0.3 * empirical_rate

Try changing the weights:
a) 0.5 * sampled + 0.5 * empirical
b) 0.3 * sampled + 0.7 * empirical  
c) 0.2 * sampled + 0.8 * empirical
""")

# Quick recommendations based on test case
print("\n" + "="*60)
print("QUICK RECOMMENDATIONS based on test case characteristics:")

if test_num == "1":
    print("""
For Test 1 (uniform prices, dominant podcast):
- Short exploration: 15-20 rounds
- Optimistic priors: alpha=2.0, beta=0.5
- Heavy empirical weight: 0.8
- Focus on identifying podcast 1 quickly
""")
elif test_num == "2":
    print("""
For Test 2 (expensive vs cheap podcasts):
- Moderate exploration: 30-40 rounds
- Balanced priors: alpha=1.5, beta=1.0
- Moderate empirical weight: 0.6
- Focus on budget efficiency
""")
else:
    print("""
For Test 3 (mixed prices):
- Moderate exploration: 25-35 rounds
- Slightly optimistic: alpha=1.5, beta=0.75
- Balanced empirical weight: 0.7
- Balance exploration with exploitation
""")

print("\n" + "="*60)
print("DEBUGGING TIPS:")
print("1. Add print statements to see which podcasts are being selected")
print("2. Track when your algorithm identifies the best podcasts")
print("3. Monitor the exploration/exploitation balance")
print("4. Check if budget is being fully utilized")

# Create a debug version
print("\nWant to see a debug version of your code? (y/n)")
if input().lower() == 'y':
    print("""
Add these debug prints to your recommend() method:

# After selecting podcasts:
if self.current_round % 100 == 0:
    print(f"Round {self.current_round}: Selected podcasts {available_podcasts}")
    
# After calculating scores:
if self.current_round == 50:
    avg_scores = np.mean(sampled_probs, axis=0)
    print(f"Average scores at round 50: {avg_scores}")
    
# In update method:
if self.current_round % 500 == 0:
    global_success = np.sum(self.successes) / np.sum(self.trials)
    print(f"Round {self.current_round}: Global success rate = {global_success:.3f}")
""")