import numpy as np
from scipy.stats import beta


class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget
        self.n_items = len(prices)
        
        # Initialize Beta distributions - start with slightly optimistic prior
        self.alpha = np.ones((n_users, self.n_items)) * 1.5  # More optimistic
        self.beta = np.ones((n_users, self.n_items)) * 0.5   # Less pessimistic
        
        # Track trials and successes
        self.trials = np.zeros((n_users, self.n_items))
        self.successes = np.zeros((n_users, self.n_items))
        
        # Current round counter
        self.current_round = 0
        
        # Precompute some useful values
        self.min_price = np.min(prices)
        self.max_podcasts_per_round = budget // self.min_price
        
        # Track global podcast performance
        self.global_successes = np.zeros(self.n_items)
        self.global_trials = np.zeros(self.n_items)
        
    def recommend(self) -> np.array:
        # Decide between exploration and exploitation
        if self.current_round < 30:
            # Pure exploration phase - try everything systematically
            return self._exploration_phase()
        
        # Sample from Beta distributions
        sampled_probs = np.zeros((self.n_users, self.n_items))
        
        # Use Thompson Sampling with empirical boost
        for i in range(self.n_users):
            for j in range(self.n_items):
                # Thompson sampling
                sampled_probs[i, j] = beta.rvs(self.alpha[i, j], self.beta[i, j])
                
                # Boost based on empirical performance if we have enough data
                if self.trials[i, j] >= 10:
                    empirical_rate = self.successes[i, j] / self.trials[i, j]
                    # Blend sampled and empirical
                    sampled_probs[i, j] = 0.7 * sampled_probs[i, j] + 0.3 * empirical_rate
        
        # Select podcasts to produce
        available_podcasts = self._select_podcasts_to_produce(sampled_probs)
        
        # Recommend for each user
        recommendations = np.zeros(self.n_users, dtype=int)
        for user in range(self.n_users):
            recommendations[user] = self._recommend_for_user(user, sampled_probs[user], available_podcasts)
        
        # Store for update
        self._last_recommendations = recommendations.copy()
        self.current_round += 1
        
        return recommendations
    
    def _exploration_phase(self):
        """Systematic exploration in early rounds"""
        recommendations = np.zeros(self.n_users, dtype=int)
        
        # Calculate which podcasts to explore
        min_trials = np.min(self.trials, axis=0)
        least_explored = np.where(min_trials == np.min(min_trials))[0]
        
        # Select podcasts that fit in budget
        selected = []
        remaining_budget = self.budget
        
        # First, add least explored podcasts
        for idx in least_explored:
            if self.item_prices[idx] <= remaining_budget:
                selected.append(idx)
                remaining_budget -= self.item_prices[idx]
                if remaining_budget < self.min_price:
                    break
        
        # Fill remaining budget with any affordable podcasts
        if remaining_budget >= self.min_price:
            for idx in range(self.n_items):
                if idx not in selected and self.item_prices[idx] <= remaining_budget:
                    selected.append(idx)
                    remaining_budget -= self.item_prices[idx]
                    if remaining_budget < self.min_price:
                        break
        
        # Assign users to podcasts round-robin style for exploration
        selected = list(selected)
        for i, user in enumerate(range(self.n_users)):
            recommendations[user] = selected[i % len(selected)]
        
        self._last_recommendations = recommendations.copy()
        self.current_round += 1
        return recommendations
    
    def _select_podcasts_to_produce(self, sampled_probs):
        """Select podcasts within budget using improved heuristics"""
        # Calculate various metrics
        expected_rewards = np.mean(sampled_probs, axis=0)
        max_rewards = np.max(sampled_probs, axis=0)
        
        # Global performance metric
        global_performance = np.zeros(self.n_items)
        for j in range(self.n_items):
            if self.global_trials[j] > 0:
                global_performance[j] = self.global_successes[j] / self.global_trials[j]
        
        # Combined score
        if self.current_round < 100:
            # Early: focus on exploration and potential
            scores = expected_rewards + 0.5 * max_rewards
        elif self.current_round < 500:
            # Middle: balance global and individual performance
            scores = 0.5 * expected_rewards + 0.3 * global_performance + 0.2 * max_rewards
        else:
            # Late: heavy focus on proven performance
            scores = 0.4 * expected_rewards + 0.5 * global_performance + 0.1 * max_rewards
        
        # Greedy selection with efficiency consideration
        selected = []
        remaining_budget = self.budget
        
        # Sort by score/price ratio (efficiency)
        efficiency = scores / np.sqrt(self.item_prices)  # Square root to not penalize expensive items too much
        sorted_indices = np.argsort(efficiency)[::-1]
        
        for idx in sorted_indices:
            if self.item_prices[idx] <= remaining_budget:
                selected.append(idx)
                remaining_budget -= self.item_prices[idx]
        
        # Ensure we produce something
        if not selected:
            cheapest = np.argmin(self.item_prices)
            if self.item_prices[cheapest] <= self.budget:
                selected = [cheapest]
        
        return set(selected)
    
    def _recommend_for_user(self, user, user_probs, available_podcasts):
        """Recommend best available podcast for a user"""
        scores = np.full(self.n_items, -np.inf)
        
        for j in available_podcasts:
            # Base score
            scores[j] = user_probs[j]
            
            # Add small exploration bonus only for very unexplored items
            if self.trials[user, j] < 5 and self.current_round < 200:
                scores[j] += 0.1 / (self.trials[user, j] + 1)
            
            # Boost items with proven high success rate
            if self.trials[user, j] >= 20:
                empirical_rate = self.successes[user, j] / self.trials[user, j]
                if empirical_rate > 0.7:  # High performers get extra boost
                    scores[j] += 0.1
        
        return np.argmax(scores)
    
    def update(self, results: np.array):
        """Update Beta distributions based on feedback"""
        for user in range(self.n_users):
            podcast = self._last_recommendations[user]
            
            # Update statistics
            self.trials[user, podcast] += 1
            self.global_trials[podcast] += 1
            
            if results[user] == 1:
                self.successes[user, podcast] += 1
                self.global_successes[podcast] += 1
                self.alpha[user, podcast] += 1
            else:
                self.beta[user, podcast] += 1