import numpy as np
from Recommender import Recommender
from simulation import Simulation
from test import tests, required_results


def test_algorithm_variant(algorithm_name, test_case, **kwargs):
    """Test a specific algorithm variant"""
    class TestRecommender(Recommender):
        def __init__(self, n_weeks, n_users, prices, budget):
            super().__init__(n_weeks, n_users, prices, budget)
            self.algorithm = algorithm_name
            # Apply any additional parameters
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Replace the Recommender class temporarily
    import simulation
    original_recommender = simulation.Recommender
    simulation.Recommender = TestRecommender
    
    # Run simulation
    sim = Simulation(test_case['P'], test_case['item_prices'], 
                     test_case['budget'], test_case['n_weeks'])
    reward = sim.simulate()
    
    # Restore original
    simulation.Recommender = original_recommender
    
    return reward


def compare_algorithms():
    """Compare different algorithm configurations"""
    algorithms_to_test = [
        ('hybrid', {}),
        ('thompson', {}),
        ('ucb', {'ucb_c': 1.0}),
        ('ucb', {'ucb_c': 2.0}),
        ('ucb', {'ucb_c': 3.0}),
        ('hybrid', {'exploration_rounds': 30}),
        ('hybrid', {'exploration_rounds': 100}),
        ('greedy', {}),
    ]
    
    print("Algorithm Performance Comparison")
    print("=" * 60)
    
    for test_idx, (test_case, required) in enumerate(zip(tests, required_results)):
        print(f"\nTest Case {test_idx + 1} (Required: {required})")
        print("-" * 40)
        
        best_reward = 0
        best_config = None
        
        for algo_name, params in algorithms_to_test:
            # Run multiple times for algorithms with randomness
            runs = 3 if algo_name in ['thompson', 'hybrid'] else 1
            rewards = []
            
            for _ in range(runs):
                reward = test_algorithm_variant(algo_name, test_case, **params)
                rewards.append(reward)
            
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards) if runs > 1 else 0
            
            param_str = f" {params}" if params else ""
            print(f"{algo_name}{param_str}: {avg_reward:.0f} ± {std_reward:.0f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_config = (algo_name, params)
        
        print(f"\nBest: {best_config[0]} with reward {best_reward:.0f}")
        print(f"Exceeds requirement: {'YES' if best_reward >= required else 'NO'}")


def analyze_test_cases():
    """Analyze the characteristics of each test case"""
    print("\nTest Case Analysis")
    print("=" * 60)
    
    for idx, test in enumerate(tests):
        print(f"\nTest {idx + 1}:")
        print(f"  Users: {test['P'].shape[0]}")
        print(f"  Items: {test['P'].shape[1]}")
        print(f"  Budget: {test['budget']}")
        print(f"  Weeks: {test['n_weeks']}")
        print(f"  Prices: {test['item_prices']}")
        print(f"  Items per week (max): {test['budget'] // np.min(test['item_prices'])}")
        print(f"  Avg probability: {np.mean(test['P']):.3f}")
        print(f"  Max probabilities by item: {np.max(test['P'], axis=0)}")


if __name__ == "__main__":
    # First analyze test cases
    analyze_test_cases()
    
    # Then compare algorithms
    print("\n" * 2)
    compare_algorithms()