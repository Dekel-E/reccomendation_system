# quick_test_fixed.py
import subprocess
import time
import os

print("Quick Automated Testing")
print("="*60)

# Get current directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Check if simulation.py exists
if not os.path.exists("simulation.py"):
    print("ERROR: simulation.py not found in current directory!")
    print("Make sure you run this script from the same folder as simulation.py")
    print("\nNavigate to the correct folder:")
    print("cd D:\\Coding\\reccomendation_system\\stochastic_mab_recsys")
    exit(1)

print("Files found. Starting tests...\n")

def run_simulation(description):
    """Run simulation and return the reward"""
    print(f"\nTesting: {description}")
    print("Running simulation.py...")
    
    # Run simulation.py using full Python path
    import sys
    python_path = sys.executable
    
    result = subprocess.run([python_path, "simulation.py"], 
                          capture_output=True, text=True)
    
    # Extract reward from output
    output = result.stdout
    if "Reward =" in output:
        reward_line = output.split("Reward =")[-1].strip()
        try:
            reward = int(reward_line)
            print(f"Result: {reward}")
            return reward
        except:
            print("Could not parse reward from:", reward_line)
            print("Full output:", output)
            return 0
    else:
        print("Error running simulation")
        if result.stderr:
            print("Error:", result.stderr)
        if result.stdout:
            print("Output:", result.stdout)
        return 0

# Test different configurations
print("\nSTEP 1: Baseline")
input("Press Enter to run baseline test...")
baseline = run_simulation("Baseline configuration")

print("\n" + "="*60)
print("STEP 2: Test exploration periods")
print("\nNow modify your Recommender.py:")
print("Change: if self.current_round < X:")
print("Where X = 20")
input("\nPress Enter after making the change...")
score_20 = run_simulation("Exploration period = 20")

print("\nChange X to 30")
input("Press Enter after making the change...")
score_30 = run_simulation("Exploration period = 30")

print("\nChange X to 15")
input("Press Enter after making the change...")
score_15 = run_simulation("Exploration period = 15")

# Find best
scores = {'15': score_15, '20': score_20, '30': score_30}
best_exploration = max(scores, key=scores.get)
print(f"\nBest exploration period: {best_exploration} (score: {scores[best_exploration]})")

print("\n" + "="*60)
print("STEP 3: Test initial priors")
print(f"\nKeep exploration at {best_exploration}")
print("Now test alpha/beta initialization:")

print("\nSet: alpha = 2.0, beta = 0.5")
input("Press Enter after making the change...")
score_optimistic = run_simulation("Optimistic priors (2.0, 0.5)")

print("\nSet: alpha = 1.5, beta = 1.0")
input("Press Enter after making the change...")
score_balanced = run_simulation("Balanced priors (1.5, 1.0)")

# Summary
print("\n" + "="*60)
print("SUMMARY OF RESULTS:")
print(f"Baseline: {baseline}")
print(f"Exploration 15: {score_15}")
print(f"Exploration 20: {score_20}")
print(f"Exploration 30: {score_30}")
print(f"Optimistic priors: {score_optimistic}")
print(f"Balanced priors: {score_balanced}")

best_score = max(baseline, score_15, score_20, score_30, score_optimistic, score_balanced)
print(f"\nBest score achieved: {best_score}")

if best_score >= 7250:
    print("✓ Congratulations! You've passed the threshold!")
else:
    print(f"Still {7250 - best_score} points short. Try more aggressive parameters.")