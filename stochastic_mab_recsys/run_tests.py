import time
from simulation import Simulation
from test_2 import all_tests # Import the list of tests
import numpy as np

def main():
    """
    Runs simulations for all test cases defined in test_cases.py.
    """
    print("--- Starting All Test Simulations ---")
    
    for i, test_config in enumerate(all_tests):
        print("\n" + "="*50)
        print(f"RUNNING TEST {i+1}: {test_config['name']}")
        print("="*50)

        # Instantiate the simulation with the current test's configuration
        simulation = Simulation(
            P=test_config['P'],
            prices=test_config['item_prices'],
            budget=test_config['budget'],
            n_weeks=test_config['n_weeks']
        )
        
        # Run the simulation
        start_time = time.perf_counter()
        final_reward = simulation.simulate()
        end_time = time.perf_counter()
        
        # Print results
        print(f"\n--- RESULTS FOR: {test_config['name']} ---")
        print(f"Final Reward: {final_reward}")
        if 'theoretical_limit' in test_config:
            print(f"Theoretical Limit: {test_config['theoretical_limit']}")
            # Calculate performance relative to the theoretical max
            performance = (final_reward / test_config['theoretical_limit']) * 100 if test_config['theoretical_limit'] > 0 else 0
            print(f"Performance vs Limit: {performance:.2f}%")
        
        print(f"Total Simulation Time: {end_time - start_time:.2f} seconds")
        print("="*50)

    print("\n--- All Simulations Finished ---")

if __name__ == '__main__':
    # This block will be executed when the script is run directly
    main()
