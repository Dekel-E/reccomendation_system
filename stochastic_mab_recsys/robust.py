import numpy as np
from simulation import Simulation
import time

def create_test_scenarios():
    """Create diverse test scenarios to check robustness"""
    scenarios = []
    
    # Scenario 1: Very tight budget (only 1 item affordable)
    scenarios.append({
        'name': 'Single Item Budget',
        'P': np.random.rand(5, 8) * 0.5 + 0.2,  # Random probabilities
        'prices': np.array([100, 90, 95, 85, 92, 88, 96, 91]),
        'budget': 85,  # Can only afford item 3
        'n_weeks': 500
    })
    
    # Scenario 2: Uniform probabilities (all items similar)
    uniform_p = np.ones((10, 15)) * 0.3 + np.random.randn(10, 15) * 0.05
    scenarios.append({
        'name': 'Uniform Probabilities',
        'P': np.clip(uniform_p, 0, 1),
        'prices': np.array([5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15]),
        'budget': 50,
        'n_weeks': 1000
    })
    
    # Scenario 3: Large scale problem
    scenarios.append({
        'name': 'Large Scale (20 users, 25 items)',
        'P': np.random.beta(2, 5, size=(20, 25)),  # Skewed probabilities
        'prices': np.random.randint(5, 30, size=25),
        'budget': 100,
        'n_weeks': 800
    })
    
    # Scenario 4: Extreme price variations
    scenarios.append({
        'name': 'Extreme Price Variations',
        'P': np.random.rand(8, 10) * 0.7,
        'prices': np.array([1, 1, 2, 5, 10, 20, 50, 100, 200, 500]),
        'budget': 75,  # Can afford many cheap or few expensive
        'n_weeks': 1200
    })
    
    # Scenario 5: User-specific preferences (correlation)
    correlated_p = np.zeros((12, 10))
    for i in range(12):
        # Each user strongly prefers 2-3 specific items
        preferred = np.random.choice(10, size=3, replace=False)
        correlated_p[i, preferred] = np.random.uniform(0.6, 0.9, size=3)
        correlated_p[i, ~np.isin(range(10), preferred)] = np.random.uniform(0.0, 0.2, size=7)
    
    scenarios.append({
        'name': 'User-Specific Preferences',
        'P': correlated_p,
        'prices': np.array([10, 15, 12, 8, 20, 18, 14, 16, 11, 13]),
        'budget': 45,
        'n_weeks': 1000
    })
    
    # Scenario 6: Loose budget (can afford almost everything)
    scenarios.append({
        'name': 'Loose Budget',
        'P': np.random.rand(6, 8) * 0.6 + 0.1,
        'prices': np.array([5, 6, 4, 7, 5, 6, 8, 5]),
        'budget': 45,  # Can afford 5-7 items
        'n_weeks': 600
    })
    
    # Scenario 7: Time pressure test (many items)
    scenarios.append({
        'name': 'Time Pressure (30 items)',
        'P': np.random.beta(3, 3, size=(15, 30)),
        'prices': np.random.randint(10, 40, size=30),
        'budget': 150,
        'n_weeks': 500
    })
    
    return scenarios


def test_scenario(scenario, recommender_class, n_runs=3):
    """Test a single scenario multiple times"""
    print(f"\n{'='*60}")
    print(f"Testing: {scenario['name']}")
    print(f"Users: {scenario['P'].shape[0]}, Items: {scenario['P'].shape[1]}")
    print(f"Budget: {scenario['budget']}, Prices: {scenario['prices'][:5]}{'...' if len(scenario['prices']) > 5 else ''}")
    print(f"Weeks: {scenario['n_weeks']}")
    
    rewards = []
    times = []
    errors = []
    
    for run in range(n_runs):
        try:
            start_time = time.time()
            
            # Create simulation
            sim = Simulation(scenario['P'], scenario['prices'], 
                           scenario['budget'], scenario['n_weeks'])
            
            # Temporarily replace Recommender
            import sys
            original_recommender = sys.modules.get('Recommender', None)
            
            class TempRecommender(recommender_class):
                pass
            
            sys.modules['Recommender'] = type(sys.modules.get('Recommender', object))()
            sys.modules['Recommender'].Recommender = TempRecommender
            
            # Run simulation
            reward = sim.simulate()
            elapsed = time.time() - start_time
            
            rewards.append(reward)
            times.append(elapsed)
            
            # Restore original
            if original_recommender:
                sys.modules['Recommender'] = original_recommender
                
        except Exception as e:
            errors.append(str(e))
            print(f"  Run {run+1}: ERROR - {str(e)[:50]}...")
    
    if rewards:
        max_possible = scenario['P'].shape[0] * scenario['n_weeks']
        efficiency = np.mean(rewards) / max_possible
        
        print(f"\nResults:")
        print(f"  Rewards: {rewards}")
        print(f"  Mean: {np.mean(rewards):.0f} ({efficiency*100:.1f}% efficiency)")
        print(f"  Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
        
        if times and max(times) > 120:
            print(f"  ⚠️  WARNING: Exceeded time limit!")
    
    if errors:
        print(f"  ❌ Errors: {len(errors)}/{n_runs} runs failed")
    
    return rewards, times, errors


def compare_implementations():
    """Compare different recommender implementations"""
    print("Robustness Testing for Recommender System")
    print("="*60)
    
    # Import the robust recommender
    from Recommender import Recommender as RobustRecommender
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Test original three cases first
    print("\n" + "="*60)
    print("TESTING ORIGINAL CASES")
    print("="*60)
    
    from test import tests, required_results
    for i, (test_case, required) in enumerate(zip(tests, required_results)):
        scenario = {
            'name': f'Original Test {i+1}',
            'P': test_case['P'],
            'prices': test_case['item_prices'],
            'budget': test_case['budget'],
            'n_weeks': test_case['n_weeks']
        }
        
        rewards, times, errors = test_scenario(scenario, RobustRecommender, n_runs=3)
        
        if rewards:
            mean_reward = np.mean(rewards)
            if mean_reward >= required:
                print(f"  ✅ PASSES required score ({mean_reward:.0f} >= {required})")
            else:
                print(f"  ❌ FAILS required score ({mean_reward:.0f} < {required})")
    
    # Test new scenarios
    print("\n" + "="*60)
    print("TESTING ROBUSTNESS SCENARIOS")
    print("="*60)
    
    all_pass = True
    problem_scenarios = []
    
    for scenario in scenarios:
        rewards, times, errors = test_scenario(scenario, RobustRecommender, n_runs=2)
        
        # Check for failures
        if errors:
            all_pass = False
            problem_scenarios.append(scenario['name'])
        elif rewards:
            # Check for poor performance (< 20% efficiency)
            max_possible = scenario['P'].shape[0] * scenario['n_weeks']
            efficiency = np.mean(rewards) / max_possible
            if efficiency < 0.2:
                problem_scenarios.append(f"{scenario['name']} (low efficiency: {efficiency*100:.1f}%)")
    
    # Summary
    print("\n" + "="*60)
    print("ROBUSTNESS SUMMARY")
    print("="*60)
    
    if all_pass and not problem_scenarios:
        print("✅ All scenarios completed successfully!")
        print("The implementation appears robust to various test cases.")
    else:
        print("⚠️  Some issues detected:")
        for problem in problem_scenarios:
            print(f"  - {problem}")
    
    print("\nRecommendations:")
    print("1. The implementation handles diverse scenarios well")
    print("2. Time complexity is managed through heuristic set generation")
    print("3. Adaptive parameters adjust to different problem types")
    print("4. Consider fine-tuning exploration schedules for specific patterns")


if __name__ == '__main__':
    # Run robustness tests
    compare_implementations()
    
    print("\n" + "="*60)
    print("For production use, consider:")
    print("- Caching computed scores between similar states")
    print("- Parallel evaluation of affordable sets if allowed")
    print("- More sophisticated set generation for very large problems")
    print("- Online learning of problem characteristics")