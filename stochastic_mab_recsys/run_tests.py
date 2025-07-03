import numpy as np
import time
from test import tests, required_results
from simulation import Simulation


class ThompsonSamplingRecommender:
    """Thompson Sampling with Beta priors"""
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int, 
                 alpha_init=1.5, beta_init=0.5, exploration_weight=0.3, exploration_decay_round=50):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget
        self.n_items = len(prices)
        
        # Thompson Sampling parameters
        self.alpha = np.ones((n_users, self.n_items)) * alpha_init
        self.beta_param = np.ones((n_users, self.n_items)) * beta_init
        
        # Track trials
        self.trials = np.zeros((n_users, self.n_items))
        
        # Parameters
        self.exploration_weight = exploration_weight
        self.exploration_decay_round = exploration_decay_round
        
        # State
        self.round = 0
        self.last_recommendations = None
        
        # Precompute affordable sets
        self._precompute_affordable_sets()
        
    def _precompute_affordable_sets(self):
        self.affordable_sets = []
        for mask in range(1, 2**self.n_items):
            items = [i for i in range(self.n_items) if mask & (1 << i)]
            if sum(self.item_prices[items]) <= self.budget:
                self.affordable_sets.append(np.array(items))
    
    def recommend(self) -> np.array:
        # Sample from posterior
        sampled_probs = np.random.beta(self.alpha, self.beta_param)
        
        # Add exploration bonus
        exploration_bonus = 1.0 / (1.0 + self.trials)
        
        # Decay exploration over time
        if self.round < self.exploration_decay_round:
            scores = sampled_probs + self.exploration_weight * exploration_bonus
        else:
            scores = sampled_probs + (self.exploration_weight * 0.3) * exploration_bonus
        
        # Find best affordable set
        best_value = -1
        best_assignments = None
        
        for item_set in self.affordable_sets:
            set_scores = scores[:, item_set]
            assignments = item_set[np.argmax(set_scores, axis=1)]
            expected_value = np.sum(scores[np.arange(self.n_users), assignments])
            
            if expected_value > best_value:
                best_value = expected_value
                best_assignments = assignments
        
        self.round += 1
        self.last_recommendations = best_assignments
        return best_assignments
    
    def update(self, results: np.array):
        if self.last_recommendations is None:
            return
            
        for user in range(self.n_users):
            item = self.last_recommendations[user]
            self.trials[user, item] += 1
            
            if results[user] == 1:
                self.alpha[user, item] += 1
            else:
                self.beta_param[user, item] += 1


class UCBRecommender:
    """Upper Confidence Bound approach"""
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int, c=1.0):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget
        self.n_items = len(prices)
        
        # UCB parameters
        self.c = c  # exploration parameter
        
        # Track successes and trials
        self.successes = np.zeros((n_users, self.n_items))
        self.trials = np.zeros((n_users, self.n_items))
        
        # State
        self.round = 0
        self.last_recommendations = None
        
        # Precompute affordable sets
        self._precompute_affordable_sets()
        
    def _precompute_affordable_sets(self):
        self.affordable_sets = []
        for mask in range(1, 2**self.n_items):
            items = [i for i in range(self.n_items) if mask & (1 << i)]
            if sum(self.item_prices[items]) <= self.budget:
                self.affordable_sets.append(np.array(items))
    
    def recommend(self) -> np.array:
        # Calculate UCB scores
        mean_rewards = np.divide(self.successes, self.trials, 
                                out=np.ones_like(self.successes), where=self.trials!=0)
        
        # Add confidence bound
        confidence = self.c * np.sqrt(np.log(self.round + 1) / (self.trials + 1))
        ucb_scores = mean_rewards + confidence
        
        # Find best affordable set
        best_value = -1
        best_assignments = None
        
        for item_set in self.affordable_sets:
            set_scores = ucb_scores[:, item_set]
            assignments = item_set[np.argmax(set_scores, axis=1)]
            expected_value = np.sum(ucb_scores[np.arange(self.n_users), assignments])
            
            if expected_value > best_value:
                best_value = expected_value
                best_assignments = assignments
        
        self.round += 1
        self.last_recommendations = best_assignments
        return best_assignments
    
    def update(self, results: np.array):
        if self.last_recommendations is None:
            return
            
        for user in range(self.n_users):
            item = self.last_recommendations[user]
            self.trials[user, item] += 1
            if results[user] == 1:
                self.successes[user, item] += 1


class EpsilonGreedyRecommender:
    """Epsilon-greedy with decay"""
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int, 
                 epsilon_start=0.3, epsilon_end=0.01, decay_rounds=200):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget
        self.n_items = len(prices)
        
        # Epsilon parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rounds = decay_rounds
        
        # Track successes and trials
        self.successes = np.zeros((n_users, self.n_items))
        self.trials = np.zeros((n_users, self.n_items))
        
        # State
        self.round = 0
        self.last_recommendations = None
        
        # Precompute affordable sets
        self._precompute_affordable_sets()
        
    def _precompute_affordable_sets(self):
        self.affordable_sets = []
        for mask in range(1, 2**self.n_items):
            items = [i for i in range(self.n_items) if mask & (1 << i)]
            if sum(self.item_prices[items]) <= self.budget:
                self.affordable_sets.append(np.array(items))
    
    def recommend(self) -> np.array:
        # Calculate epsilon for current round
        if self.round < self.decay_rounds:
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.round / self.decay_rounds)
        else:
            epsilon = self.epsilon_end
        
        # Calculate estimated probabilities
        est_probs = np.divide(self.successes, self.trials, 
                             out=np.ones_like(self.successes) * 0.5, where=self.trials!=0)
        
        if np.random.random() < epsilon:
            # Explore: random selection
            item_set = np.random.choice(len(self.affordable_sets))
            selected_set = self.affordable_sets[item_set]
            assignments = np.random.choice(selected_set, size=self.n_users)
        else:
            # Exploit: greedy selection
            best_value = -1
            assignments = None
            
            for item_set in self.affordable_sets:
                set_probs = est_probs[:, item_set]
                set_assignments = item_set[np.argmax(set_probs, axis=1)]
                expected_value = np.sum(est_probs[np.arange(self.n_users), set_assignments])
                
                if expected_value > best_value:
                    best_value = expected_value
                    assignments = set_assignments
        
        self.round += 1
        self.last_recommendations = assignments
        return assignments
    
    def update(self, results: np.array):
        if self.last_recommendations is None:
            return
            
        for user in range(self.n_users):
            item = self.last_recommendations[user]
            self.trials[user, item] += 1
            if results[user] == 1:
                self.successes[user, item] += 1


def test_strategy(recommender_class, test_case, params=None, n_runs=5):
    """Test a strategy multiple times and return average reward"""
    if params is None:
        params = {}
    
    rewards = []
    times = []
    
    for _ in range(n_runs):
        # Create new recommender instance
        class TestRecommender(recommender_class):
            def __init__(self, n_weeks, n_users, prices, budget):
                super().__init__(n_weeks, n_users, prices, budget, **params)
        
        # Run simulation
        start_time = time.time()
        sim = Simulation(test_case['P'], test_case['item_prices'], 
                        test_case['budget'], test_case['n_weeks'])
        
        # Temporarily replace Recommender
        import sys
        sys.modules['Recommender'].Recommender = TestRecommender
        
        reward = sim.simulate()
        elapsed = time.time() - start_time
        
        rewards.append(reward)
        times.append(elapsed)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_time': np.mean(times),
        'rewards': rewards
    }


# Test different strategies and parameters
if __name__ == '__main__':
    strategies = [
        {
            'name': 'Thompson Sampling (default)',
            'class': ThompsonSamplingRecommender,
            'params': {}
        },
        {
            'name': 'Thompson Sampling (optimistic)',
            'class': ThompsonSamplingRecommender,
            'params': {'alpha_init': 2.0, 'beta_init': 0.5}
        },
        {
            'name': 'Thompson Sampling (more exploration)',
            'class': ThompsonSamplingRecommender,
            'params': {'exploration_weight': 0.5, 'exploration_decay_round': 100}
        },
        {
            'name': 'UCB (c=1.0)',
            'class': UCBRecommender,
            'params': {'c': 1.0}
        },
        {
            'name': 'UCB (c=2.0)',
            'class': UCBRecommender,
            'params': {'c': 2.0}
        },
        {
            'name': 'Epsilon-Greedy',
            'class': EpsilonGreedyRecommender,
            'params': {}
        }
    ]
    
    print("Testing different strategies...\n")
    
    for test_idx, (test_case, required_score) in enumerate(zip(tests, required_results)):
        print(f"\n=== Test Case {test_idx + 1} (Required: {required_score}) ===")
        print(f"Users: {test_case['P'].shape[0]}, Items: {test_case['P'].shape[1]}")
        print(f"Budget: {test_case['budget']}, Weeks: {test_case['n_weeks']}")
        print(f"Item prices: {test_case['item_prices']}")
        
        best_strategy = None
        best_score = -1
        
        for strategy in strategies:
            print(f"\nTesting {strategy['name']}...")
            
            try:
                result = test_strategy(strategy['class'], test_case, 
                                     strategy['params'], n_runs=3)
                
                print(f"  Mean reward: {result['mean_reward']:.0f} ± {result['std_reward']:.0f}")
                print(f"  Mean time: {result['mean_time']:.1f}s")
                print(f"  Individual runs: {[int(r) for r in result['rewards']]}")
                
                if result['mean_reward'] > best_score:
                    best_score = result['mean_reward']
                    best_strategy = strategy['name']
                    
                if result['mean_reward'] >= required_score:
                    print(f"  ✓ PASSES required score!")
                    
            except Exception as e:
                print(f"  ERROR: {str(e)}")
        
        print(f"\nBest strategy for test {test_idx + 1}: {best_strategy} ({best_score:.0f})")
        
    print("\n\nRecommendation: Use the strategy that performs best across all test cases.")
    print("You may need to tune parameters further for the competitive portion.")