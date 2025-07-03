import numpy as np
from scipy import stats


class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.n_items = len(prices)
        self.item_prices = prices
        self.budget = budget
        
        # We chose between 3 algorithm variants: thompson sampling, UCB,  hybrid approach , and greedy
        self.algorithm = 'hybrid'  # 'thompson', 'ucb', 'hybrid', 'greedy'
        
        # Initialize statistics variables
        self.successes = np.zeros((n_users, self.n_items))
        self.failures = np.zeros((n_users, self.n_items))
        self.total_pulls = np.zeros((n_users, self.n_items))
        
        # UCB parameters
        self.ucb_c = 2.0  # Exploration parameter
        
        # Thompson Sampling parameters
        self.alpha = np.ones((n_users, self.n_items))  # Beta dist alpha
        self.beta = np.ones((n_users, self.n_items))   # Beta dist beta
        
        # Exploration parameters
        self.round_num = 0
        self.exploration_rounds = max(30, min(100, n_weeks // 50))  # Adaptive exploration
        self.min_exploration_prob = 0.01
        
        # Performance tracking
        self.recent_rewards = []
        self.reward_window = 50
        
        # Efficiency optimizations
        self.update_frequency = 10  # Update estimates every N rounds
        self.last_estimate_update = 0
        
        # Cache for efficiency
        self.last_selected_items = None
        self.estimated_probs = np.zeros((n_users, self.n_items))
        
    def recommend(self) -> np.array:
        self.round_num += 1
        
        if self.algorithm == 'thompson':
            return self._thompson_sampling_recommend()
        elif self.algorithm == 'ucb':
            return self._ucb_recommend()
        elif self.algorithm == 'hybrid':
            return self._hybrid_recommend()
        else:
            return self._greedy_recommend()
    
    def _thompson_sampling_recommend(self):
        """Thompson Sampling with smart budget allocation"""
        # Sample from Beta distributions
        sampled_probs = np.random.beta(self.alpha, self.beta)
        
        # Select items within budget
        selected_items = self._select_items_knapsack(sampled_probs)
        
        # Make recommendations
        recommendations = self._assign_items_to_users(sampled_probs, selected_items)
        
        return recommendations
    
    def _ucb_recommend(self):
        # Calculate UCB scores
        avg_rewards = self.successes / (self.total_pulls + 1e-10)
        
        # Adaptive exploration parameter
        exploration_bonus = self.ucb_c * np.sqrt(
            np.log(self.round_num + 1) / (self.total_pulls + 1)
        )
        
        ucb_scores = avg_rewards + exploration_bonus
        
        # Select items within budget
        selected_items = self._select_items_knapsack(ucb_scores)
        
        # Make recommendations
        recommendations = self._assign_items_to_users(ucb_scores, selected_items)
        
        return recommendations
    
    def _hybrid_recommend(self):
        # Combine Thompson Sampling with smart budget allocation
        
        # Early exploration phase
        if self.round_num <= self.exploration_rounds:
            return self._exploration_phase()
        
        # Sample from posterior
        sampled_probs = np.random.beta(self.alpha, self.beta)
        
        # Also calculate mean estimates for stability
        mean_probs = self.alpha / (self.alpha + self.beta)
        
        # Adaptive exploration based on performance
        if len(self.recent_rewards) >= self.reward_window:
            recent_avg = np.mean(self.recent_rewards[-self.reward_window:])
            older_avg = np.mean(self.recent_rewards[:-self.reward_window]) if len(self.recent_rewards) > self.reward_window else 0
            
            # If performance is declining, increase exploration
            if recent_avg < older_avg * 0.95:
                exploration_boost = 0.2
            else:
                exploration_boost = 0.0
        else:
            exploration_boost = 0.1
        
        # Blend sampled and mean probabilities (more exploration early on)
        blend_factor = max(self.min_exploration_prob, 
                          min(0.5, (1.0 - self.round_num / self.n_rounds) + exploration_boost))
        blended_probs = blend_factor * sampled_probs + (1 - blend_factor) * mean_probs
        
        # Update cached probability estimates periodically
        if self.round_num - self.last_estimate_update >= self.update_frequency:
            self.estimated_probs = mean_probs
            self.last_estimate_update = self.round_num
        
        # Select items using expected value optimization
        selected_items = self._select_items_advanced(blended_probs, mean_probs)
        
        # Make recommendations
        recommendations = self._assign_items_to_users(blended_probs, selected_items)
        
        return recommendations
    
    def _greedy_recommend(self):
        # Simple greedy with epsilon-greedy exploration
        epsilon = max(self.min_exploration_prob, 
                     1.0 - self.round_num / self.exploration_rounds)
        
        if np.random.random() < epsilon:
            # Explore
            return self._random_valid_recommendation()
        else:
            # Exploit
            mean_probs = self.successes / (self.total_pulls + 1e-10)
            selected_items = self._select_items_knapsack(mean_probs)
            recommendations = self._assign_items_to_users(mean_probs, selected_items)
            return recommendations
    
    def _exploration_phase(self):
        """Smart exploration in early rounds"""
        # Systematic exploration with budget awareness
        items_to_explore = []
        
        # Calculate exploration priority
        total_item_pulls = np.sum(self.total_pulls, axis=0)
        
        # Add small random noise to break ties
        exploration_priority = total_item_pulls + np.random.random(self.n_items) * 0.1
        
        # Sort items by exploration priority (least explored first)
        priority_order = np.argsort(exploration_priority)
        
        # Select items within budget, prioritizing underexplored
        current_cost = 0
        for item in priority_order:
            if current_cost + self.item_prices[item] <= self.budget:
                items_to_explore.append(item)
                current_cost += self.item_prices[item]
                
                # Stop if we've used most of the budget
                if current_cost >= self.budget * 0.9:
                    break
        
        # Ensure we have at least one item
        if not items_to_explore:
            items_to_explore = [np.argmin(self.item_prices)]
        
        # Assign to users - mix random and potential-based assignment
        if self.round_num <= self.exploration_rounds // 2:
            # Pure random in very early rounds
            recommendations = np.random.choice(items_to_explore, size=self.n_users)
        else:
            # Start using some knowledge
            current_estimates = self.alpha / (self.alpha + self.beta)
            recommendations = self._assign_items_to_users(current_estimates, items_to_explore)
        
        self.last_selected_items = items_to_explore
        return recommendations
    
    def _select_items_knapsack(self, scores):
        """Select items using knapsack-style optimization"""
        # Calculate expected value per user per item
        expected_values = np.sum(scores, axis=0)
        
        # Value per cost ratio
        value_per_cost = expected_values / self.item_prices
        
        # Sort by value per cost
        sorted_indices = np.argsort(-value_per_cost)
        
        selected_items = []
        current_cost = 0
        
        for idx in sorted_indices:
            if current_cost + self.item_prices[idx] <= self.budget:
                selected_items.append(idx)
                current_cost += self.item_prices[idx]
        
        # Ensure we have at least one item
        if not selected_items:
            cheapest_item = np.argmin(self.item_prices)
            selected_items = [cheapest_item]
        
        self.last_selected_items = selected_items
        return selected_items
    
    def _select_items_advanced(self, sampled_probs, mean_probs):
        """Advanced item selection considering uncertainty and expected value"""
        # Calculate expected value with uncertainty bonus
        expected_values = np.sum(mean_probs, axis=0)
        
        # Calculate uncertainty (variance of Beta distribution)
        uncertainty = np.sum(self.alpha * self.beta / 
                           ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1)), 
                           axis=0)
        
        # Adaptive uncertainty weight based on round
        uncertainty_weight = max(0.2, 1.0 - self.round_num / (self.n_rounds * 0.5))
        
        # Combine value and uncertainty
        selection_scores = expected_values + uncertainty_weight * np.sqrt(uncertainty)
        
        # Check if we should use greedy selection (late game)
        if self.round_num > self.n_rounds * 0.8:
            # Pure exploitation in end game
            return self._select_items_knapsack(mean_probs)
        
        # Dynamic programming approximation for subset selection
        selected_items = []
        current_cost = 0
        
        # First pass: select best items with good value/cost ratio
        value_per_cost = selection_scores / (self.item_prices + 1e-10)
        sorted_by_efficiency = np.argsort(-value_per_cost)
        
        # Reserve some budget for diversity
        efficiency_budget = self.budget * 0.7
        
        for item in sorted_by_efficiency:
            if current_cost + self.item_prices[item] <= efficiency_budget:
                selected_items.append(item)
                current_cost += self.item_prices[item]
        
        # Second pass: add diverse items
        remaining_budget = self.budget - current_cost
        remaining_items = [i for i in range(self.n_items) if i not in selected_items]
        
        if remaining_budget > 0 and remaining_items:
            # Add items with high uncertainty or underexplored
            exploration_scores = uncertainty[remaining_items] / (self.total_pulls.sum(axis=0)[remaining_items] + 1)
            sorted_by_exploration = remaining_items[np.argsort(-exploration_scores)]
            
            for item in sorted_by_exploration:
                if current_cost + self.item_prices[item] <= self.budget:
                    selected_items.append(item)
                    current_cost += self.item_prices[item]
        
        # Ensure we have at least one item
        if not selected_items:
            selected_items = [np.argmin(self.item_prices)]
        
        self.last_selected_items = selected_items
        return selected_items
    
    def _assign_items_to_users(self, scores, selected_items):
        """Assign selected items to users optimally"""
        recommendations = np.zeros(self.n_users, dtype=int)
        
        # Get scores only for selected items
        selected_scores = scores[:, selected_items]
        
        # Assign each user their best item from selected items
        for user in range(self.n_users):
            best_item_idx = np.argmax(selected_scores[user])
            recommendations[user] = selected_items[best_item_idx]
        
        return recommendations
    
    def _random_valid_recommendation(self):
        """Generate random valid recommendation for exploration"""
        # Random item selection within budget
        items = list(range(self.n_items))
        np.random.shuffle(items)
        
        selected_items = []
        current_cost = 0
        
        for item in items:
            if current_cost + self.item_prices[item] <= self.budget:
                selected_items.append(item)
                current_cost += self.item_prices[item]
                if current_cost >= self.budget * 0.8:  # Use most of budget
                    break
        
        if not selected_items:
            selected_items = [np.argmin(self.item_prices)]
        
        # Random assignment
        recommendations = np.random.choice(selected_items, size=self.n_users)
        self.last_selected_items = selected_items
        
        return recommendations
    
    def update(self, results: np.array):
        """Update statistics based on user feedback"""
        # Get recommendations from last round
        if hasattr(self, '_last_recommendations'):
            recommendations = self._last_recommendations
        else:
            # Reconstruct from results if needed
            return
        
        # Track performance
        total_reward = np.sum(results)
        self.recent_rewards.append(total_reward)
        
        # Keep window size manageable
        if len(self.recent_rewards) > self.reward_window * 3:
            self.recent_rewards = self.recent_rewards[-self.reward_window * 2:]
        
        # Update statistics
        for user in range(self.n_users):
            item = recommendations[user]
            
            if results[user] == 1:
                self.successes[user, item] += 1
                self.alpha[user, item] += 1
            else:
                self.failures[user, item] += 1
                self.beta[user, item] += 1
            
            self.total_pulls[user, item] += 1
    
    def recommend(self) -> np.array:
        """Override to save recommendations"""
        self.round_num += 1
        
        if self.algorithm == 'thompson':
            recommendations = self._thompson_sampling_recommend()
        elif self.algorithm == 'ucb':
            recommendations = self._ucb_recommend()
        elif self.algorithm == 'hybrid':
            recommendations = self._hybrid_recommend()
        else:
            recommendations = self._greedy_recommend()
        
        self._last_recommendations = recommendations
        return recommendations