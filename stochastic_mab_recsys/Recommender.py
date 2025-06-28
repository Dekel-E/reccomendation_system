import numpy as np
from scipy.stats import beta


class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget
        self.n_items = len(prices)
        
        # Initialize Beta distributions for Thompson Sampling
        # FIXED: np.ones needs a tuple for shape
        self.alpha = np.ones((n_users, self.n_items))
        self.beta = np.ones((n_users, self.n_items))
        
        # Track trials and successes
        self.trials = np.zeros((n_users, self.n_items))
        self.successes = np.zeros((n_users, self.n_items))
        
        # Current round counter
        self.current_round = 0
        
        # Precompute some useful values
        self.min_price = np.min(prices)
        self.max_podcasts_per_round = budget // self.min_price
        
    def recommend(self) -> np.array:
        # Sample from Beta distributions
        sampled_probs = np.zeros((self.n_users, self.n_items))
        for i in range(self.n_users):
            for j in range(self.n_items):
                sampled_probs[i, j] = beta.rvs(self.alpha[i, j], self.beta[i, j])
        
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
    
    def _select_podcasts_to_produce(self, sampled_probs):
        """Intelligently select podcasts within budget"""
        # Calculate metrics for each podcast
        expected_rewards = np.mean(sampled_probs, axis=0)
        max_rewards = np.max(sampled_probs, axis=0)
        
        # Production counts and exploration bonus
        production_counts = np.sum(self.trials, axis=0)
        exploration_bonus = np.sqrt(2 * np.log(self.current_round + 1) / (production_counts + 1))
        
        # UCB-style score combining different factors
        if self.current_round < 50:
            # Heavy exploration initially
            scores = expected_rewards + 2.0 * exploration_bonus
        elif self.current_round < 200:
            # Balanced exploration/exploitation
            scores = expected_rewards + exploration_bonus
        else:
            # Focus on exploitation with minimal exploration
            scores = 0.7 * expected_rewards + 0.3 * max_rewards + 0.2 * exploration_bonus
        
        # Dynamic programming for budget allocation
        selected = self._dp_budget_allocation(scores, self.item_prices, self.budget)
        
        # Ensure we produce at least something
        if not selected:
            # Fallback: select highest score items that fit budget
            sorted_indices = np.argsort(scores)[::-1]
            for idx in sorted_indices:
                if self.item_prices[idx] <= self.budget:
                    selected = [idx]
                    break
        
        return set(selected)
    
    def _dp_budget_allocation(self, scores, prices, budget):
        """Use dynamic programming for optimal budget allocation"""
        n = len(scores)
        # Scale scores to avoid numerical issues
        scaled_scores = scores * 100
        
        # DP table: dp[i][j] = (max_score, items) using first i items with budget j
        dp = {}
        dp[0] = (0, [])
        
        for i in range(n):
            new_dp = {}
            for b, (score, items) in dp.items():
                # Don't take item i
                if b not in new_dp or new_dp[b][0] < score:
                    new_dp[b] = (score, items)
                
                # Take item i if budget allows
                if b + prices[i] <= budget:
                    new_score = score + scaled_scores[i]
                    new_budget = b + prices[i]
                    if new_budget not in new_dp or new_dp[new_budget][0] < new_score:
                        new_dp[new_budget] = (new_score, items + [i])
            dp = new_dp
        
        # Find best allocation
        best_score = -1
        best_items = []
        for b, (score, items) in dp.items():
            if score > best_score:
                best_score = score
                best_items = items
        
        return best_items
    
    def _recommend_for_user(self, user, user_probs, available_podcasts):
        """Recommend best available podcast for a user"""
        # Create scores for available podcasts
        scores = np.full(self.n_items, -np.inf)
        
        for j in available_podcasts:
            # Base score is sampled probability
            scores[j] = user_probs[j]
            
            # Add exploration bonus for rarely tried podcasts
            if self.current_round < 100 and self.trials[user, j] < 2:
                scores[j] += 0.3 / (self.trials[user, j] + 1)
            elif self.current_round < 300 and self.trials[user, j] < 5:
                scores[j] += 0.1 / (self.trials[user, j] + 1)
        
        return np.argmax(scores)
    
    def update(self, results: np.array):
        """Update Beta distributions based on feedback"""
        for user in range(self.n_users):
            podcast = self._last_recommendations[user]
            
            # Update statistics
            self.trials[user, podcast] += 1
            
            if results[user] == 1:
                self.successes[user, podcast] += 1
                self.alpha[user, podcast] += 1
            else:
                self.beta[user, podcast] += 1