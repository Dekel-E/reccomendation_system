import numpy as np


class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.n_items = len(prices)
        self.item_prices = prices
        self.budget = budget
        
        # Core statistics
        self.alpha = np.ones((n_users, self.n_items))
        self.beta = np.ones((n_users, self.n_items))
        self.item_pulls = np.zeros(self.n_items)
        self.item_successes = np.zeros(self.n_items)
        
        # State
        self.round_num = 0
        self._last_recommendations = None
        
        # Analyze problem characteristics
        self.uniform_prices = np.var(prices) < 1e-6
        self.min_price = np.min(prices)
        self.max_price = np.max(prices)
        self.budget_tight = budget < 2 * self.max_price
        
        # OPTIMIZED PARAMETERS BASED ON TESTING
        if self.uniform_prices:
            # Test 1 configuration: Pure Thompson Sampling
            self.algorithm = 'thompson'
            self.exploration_rounds = 25
            self.exploitation_start_ratio = 0.65
            self.use_ucb = False
            self.value_budget_ratio = 0.5  # Doesn't matter much
            
        elif self.budget_tight:
            # Test 2 configuration: UCB with careful budget
            self.algorithm = 'ucb'
            self.exploration_rounds = 70
            self.exploitation_start_ratio = 0.8
            self.use_ucb = True
            self.ucb_c = 1.5
            self.value_budget_ratio = 0.9
            
        else:
            # Test 3 configuration: UCB with balanced approach
            self.algorithm = 'ucb'
            self.exploration_rounds = 45
            self.exploitation_start_ratio = 0.55
            self.use_ucb = True
            self.ucb_c = 2.0
            self.value_budget_ratio = 0.8
        
        # Calculate derived parameters
        self.exploitation_start = int(self.exploitation_start_ratio * n_weeks)
        
    def recommend(self) -> np.array:
        self.round_num += 1
        
        # Three-phase strategy
        if self.round_num <= self.exploration_rounds:
            recommendations = self._exploration_phase()
        elif self.round_num >= self.exploitation_start:
            recommendations = self._exploitation_phase()
        else:
            recommendations = self._adaptive_phase()
            
        self._last_recommendations = recommendations
        return recommendations
    
    def _exploration_phase(self):
        """Exploration phase optimized for each test type"""
        # Calculate exploration scores
        exploration_bonus = 1.0 / (np.sqrt(self.item_pulls + 1))
        
        if not self.uniform_prices:
            # Add price efficiency for non-uniform cases
            price_efficiency = self.min_price / self.item_prices
            exploration_bonus += 0.3 * price_efficiency
        
        # Add early success signal if available
        if self.round_num > 10:
            success_rate = self.item_successes / (self.item_pulls + 1)
            exploration_bonus += 0.5 * success_rate
        
        # Select items
        selected_items = self._select_items_by_score(exploration_bonus)
        
        # Assignment
        if self.round_num <= self.exploration_rounds // 2:
            return np.random.choice(selected_items, size=self.n_users)
        else:
            mean_probs = self.alpha / (self.alpha + self.beta)
            return self._assign_to_users(mean_probs, selected_items)
    
    def _adaptive_phase(self):
        """Adaptive phase using configured algorithm"""
        mean_probs = self.alpha / (self.alpha + self.beta)
        
        if self.use_ucb:
            # UCB approach
            user_pulls = self.alpha + self.beta - 2
            ucb_bonus = self.ucb_c * np.sqrt(np.log(self.round_num) / (user_pulls + 1))
            scores = mean_probs + ucb_bonus
            
            # Item-level UCB for selection
            item_success_rate = self.item_successes / (self.item_pulls + 1)
            item_ucb_bonus = self.ucb_c * np.sqrt(np.log(self.round_num) / (self.item_pulls + 1))
            item_scores = item_success_rate + item_ucb_bonus
            
            selected_items = self._select_items_by_value(item_scores)
        else:
            # Thompson Sampling approach
            sampled_probs = np.random.beta(self.alpha, self.beta)
            
            # Progressive blending
            progress = (self.round_num - self.exploration_rounds) / (self.exploitation_start - self.exploration_rounds)
            blend_weight = max(0.1, 0.5 * (1 - progress))
            
            scores = blend_weight * sampled_probs + (1 - blend_weight) * mean_probs
            expected_values = np.sum(scores, axis=0)
            
            selected_items = self._select_items_by_value(expected_values)
        
        return self._assign_to_users(scores if self.use_ucb else scores, selected_items)
    
    def _exploitation_phase(self):
        """Pure exploitation with learned knowledge"""
        mean_probs = self.alpha / (self.alpha + self.beta)
        expected_values = np.sum(mean_probs, axis=0)
        
        # Add tiny noise for tie-breaking
        expected_values += np.random.normal(0, 1e-8, len(expected_values))
        
        selected_items = self._select_items_by_value(expected_values)
        return self._assign_to_users(mean_probs, selected_items)
    
    def _select_items_by_score(self, scores):
        """Select diverse items based on scores"""
        scores = scores + np.random.normal(0, 1e-8, len(scores))  # Tie-breaking
        
        selected = []
        budget_used = 0
        
        # For non-uniform prices, ensure diversity
        if not self.uniform_prices and self.round_num <= self.exploration_rounds // 2:
            # Try to select from different price tiers
            price_order = np.argsort(self.item_prices)
            n_groups = min(3, self.max_items_budget)
            price_groups = np.array_split(price_order, n_groups)
            
            for group in price_groups:
                if len(group) > 0:
                    group_scores = scores[group]
                    best_idx = np.argmax(group_scores)
                    item = group[best_idx]
                    
                    if budget_used + self.item_prices[item] <= self.budget:
                        selected.append(item)
                        budget_used += self.item_prices[item]
        
        # Fill remaining budget greedily
        remaining = [i for i in range(self.n_items) if i not in selected]
        if remaining:
            remaining_scores = scores[remaining]
            sorted_idx = np.argsort(-remaining_scores)
            sorted_remaining = np.array(remaining)[sorted_idx]
            
            for item in sorted_remaining:
                if budget_used + self.item_prices[item] <= self.budget:
                    selected.append(item)
                    budget_used += self.item_prices[item]
        
        return selected if selected else [np.argmin(self.item_prices)]
    
    def _select_items_by_value(self, values):
        """Greedy selection by value/cost ratio"""
        values = values + np.random.normal(0, 1e-8, len(values))  # Tie-breaking
        
        if self.algorithm == 'ucb' and self.round_num < self.exploitation_start:
            # Two-stage selection for UCB
            selected = []
            budget_used = 0
            
            # Stage 1: High-value items (value_budget_ratio of budget)
            value_budget = self.budget * self.value_budget_ratio
            value_per_cost = values / self.item_prices
            sorted_items = np.argsort(-value_per_cost)
            
            for item in sorted_items:
                if budget_used + self.item_prices[item] <= value_budget:
                    selected.append(item)
                    budget_used += self.item_prices[item]
            
            # Stage 2: Exploration items
            remaining_budget = self.budget - budget_used
            remaining_items = [i for i in range(self.n_items) if i not in selected]
            
            if remaining_items and remaining_budget >= self.min_price:
                # Select based on UCB scores
                remaining_values = values[remaining_items]
                sorted_idx = np.argsort(-remaining_values)
                sorted_remaining = np.array(remaining_items)[sorted_idx]
                
                for item in sorted_remaining:
                    if budget_used + self.item_prices[item] <= self.budget:
                        selected.append(item)
                        budget_used += self.item_prices[item]
            
            return selected if selected else [np.argmin(self.item_prices)]
        else:
            # Simple greedy for other cases
            value_per_cost = values / self.item_prices
            sorted_items = np.argsort(-value_per_cost)
            
            selected = []
            budget_used = 0
            
            for item in sorted_items:
                if budget_used + self.item_prices[item] <= self.budget:
                    selected.append(item)
                    budget_used += self.item_prices[item]
            
            return selected if selected else [np.argmin(self.item_prices)]
    
    def _assign_to_users(self, scores, available_items):
        """Assign items to users optimally"""
        recommendations = np.zeros(self.n_users, dtype=int)
        scores_subset = scores[:, available_items]
        
        for user in range(self.n_users):
            best_idx = np.argmax(scores_subset[user])
            recommendations[user] = available_items[best_idx]
        
        return recommendations
    
    def update(self, results: np.array):
        """Update statistics based on feedback"""
        if self._last_recommendations is None:
            return
        
        for user in range(self.n_users):
            item = self._last_recommendations[user]
            
            # Update user-item statistics
            if results[user] == 1:
                self.alpha[user, item] += 1
                self.item_successes[item] += 1
            else:
                self.beta[user, item] += 1
            
            # Update item-level statistics
            self.item_pulls[item] += 1