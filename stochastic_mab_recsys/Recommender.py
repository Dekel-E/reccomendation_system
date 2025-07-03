import numpy as np


class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget
        self.n_items = len(prices)
        
        # Detect if we have a tight budget constraint and act accordinglt
        self.tight_budget = self._is_tight_budget()
        
        # Thompson Sampling parameters - adjust based on budget tightness
        if self.tight_budget:
            #if we run on tight budget, we want to be more conservative
            # and avoid over-exploration, so we initialize with higher alpha and lower beta
            self.alpha = np.ones((n_users, self.n_items)) * 3.0
            self.beta_param = np.ones((n_users, self.n_items)) * 0.3
        else:
            # Standard optimistic initialization
            # For normal budgets, we can afford more exploration
            self.alpha = np.ones((n_users, self.n_items)) * 2.0
            self.beta_param = np.ones((n_users, self.n_items)) * 0.5
        
        # Track statistics
        self.trials = np.zeros((n_users, self.n_items))
        self.successes = np.zeros((n_users, self.n_items))
        
        # State tracking
        self.round = 0
        self.last_recommendations = None
        
        # Precompute affordable item sets
        self._precompute_affordable_sets()
        
        # UCB parameters for hybrid approach
        self.ucb_c = 1.0 if not self.tight_budget else 0.5
        
    def _is_tight_budget(self):
        """Check if budget constraints are tight"""
        min_price = np.min(self.item_prices)
        max_affordable = self.budget // min_price
        return max_affordable <= 3 or self.budget <= np.median(self.item_prices) * 2
        
    def _precompute_affordable_sets(self):
        """Precompute all affordable podcast combinations"""
        self.affordable_sets = []
        
        # For very tight budgets, be more exhaustive
        if self.tight_budget:
            # Check all possible combinations more carefully
            for mask in range(1, min(2**self.n_items, 1024)):
                items = [i for i in range(self.n_items) if mask & (1 << i)]
                if sum(self.item_prices[items]) <= self.budget:
                    self.affordable_sets.append(np.array(items))
        else:
            # Standard approach for normal budgets
            self._generate_affordable_sets_efficiently()
    
    def _generate_affordable_sets_efficiently(self):
        """Generate affordable sets efficiently for normal budgets"""
        price_order = np.argsort(self.item_prices)
        
        # Dynamic programming approach
        for size in range(1, min(self.n_items + 1, 10)):
            self._add_sets_of_size(size, price_order)
    
    def _add_sets_of_size(self, size, price_order):
        """Add all affordable sets of given size"""
        def generate(idx, current_set, current_cost):
            if len(current_set) == size:
                if current_cost <= self.budget:
                    self.affordable_sets.append(np.array(current_set))
                return
            
            for i in range(idx, self.n_items):
                item = price_order[i]
                new_cost = current_cost + self.item_prices[item]
                if new_cost <= self.budget:
                    generate(i + 1, current_set + [item], new_cost)
                elif len(current_set) > 0:
                    break
        
        generate(0, [], 0)
    
    def recommend(self) -> np.array:
        # Use hybrid approach for tight budgets
        if self.tight_budget and self.round < 100:
            return self._recommend_hybrid()
        else:
            return self._recommend_thompson_sampling()
    
    def _recommend_hybrid(self):
        """Hybrid approach combining Thompson Sampling and UCB for tight budgets"""
        # Calculate both Thompson Sampling and UCB scores
        
        # Thompson Sampling scores
        sampled_probs = np.random.beta(self.alpha, self.beta_param)
        
        # UCB scores
        mean_rewards = np.divide(self.successes, self.trials, 
                                out=np.ones_like(self.successes) * 0.5, 
                                where=self.trials!=0)
        confidence = self.ucb_c * np.sqrt(np.log(self.round + 1) / (self.trials + 1))
        ucb_scores = mean_rewards + confidence
        
        # Combine scores - more weight on UCB early for faster convergence
        if self.round < 30:
            scores = 0.3 * sampled_probs + 0.7 * ucb_scores
        else:
            scores = 0.6 * sampled_probs + 0.4 * ucb_scores
        
        # Add small bonus for completely untried items
        untried_bonus = (self.trials == 0) * 0.2
        scores += untried_bonus
        
        return self._select_best_assignment(scores)
    
    def _recommend_thompson_sampling(self):
        """Standard Thompson Sampling approach"""
        # Sample from posterior
        sampled_probs = np.random.beta(self.alpha, self.beta_param)
        
        # Calculate variance for exploration bonus
        variance = (self.alpha * self.beta_param) / ((self.alpha + self.beta_param)**2 * 
                                                     (self.alpha + self.beta_param + 1))
        exploration_bonus = np.sqrt(variance)
        
        # Adaptive exploration weight
        if self.round < 50:
            exploration_weight = 0.4
        elif self.round < 150:
            exploration_weight = 0.2
        elif self.round < 400:
            exploration_weight = 0.1
        else:
            exploration_weight = 0.05
        
        # Combined scores
        scores = sampled_probs + exploration_weight * exploration_bonus
        
        # Bonus for untried items
        untried_bonus = (self.trials == 0) * 0.15
        scores += untried_bonus
        
        return self._select_best_assignment(scores)
    
    def _select_best_assignment(self, scores):
        """Select best assignment given scores"""
        best_value = -np.inf
        best_assignments = None
        
        for item_set in self.affordable_sets:
            # Get scores for this set
            set_scores = scores[:, item_set]
            
            # Optimal assignment
            best_items_idx = np.argmax(set_scores, axis=1)
            assignments = item_set[best_items_idx]
            
            # Calculate expected value
            expected_value = np.sum(set_scores[np.arange(self.n_users), best_items_idx])
            
            # For tight budgets, add stability bonus to reduce switching
            if self.tight_budget and self.round > 50:
                if self.last_recommendations is not None:
                    stability_bonus = np.sum(assignments == self.last_recommendations) * 0.01
                    expected_value += stability_bonus
            
            if expected_value > best_value:
                best_value = expected_value
                best_assignments = assignments
        
        self.round += 1
        self.last_recommendations = best_assignments
        
        return best_assignments
    
    def update(self, results: np.array):
        """Update beliefs based on observed results"""
        if self.last_recommendations is None:
            return
        
        # Vectorized update
        user_indices = np.arange(self.n_users)
        item_indices = self.last_recommendations
        
        # Update trials and successes
        self.trials[user_indices, item_indices] += 1
        self.successes[user_indices, item_indices] += results
        
        # Update Beta parameters
        self.alpha[user_indices, item_indices] += results
        self.beta_param[user_indices, item_indices] += (1 - results)
        
        # For tight budgets, boost learning for high-performing items
        if self.tight_budget and self.round > 20:
            # Identify items with high success rates
            success_rates = np.divide(self.successes, self.trials, 
                                    out=np.zeros_like(self.successes), 
                                    where=self.trials > 5)
            
            # Boost confidence for consistently good items
            for user in range(self.n_users):
                for item in range(self.n_items):
                    if success_rates[user, item] > 0.7 and self.trials[user, item] > 10:
                        # Increase confidence by reducing uncertainty
                        boost = 0.1
                        self.alpha[user, item] *= (1 + boost)
                        self.beta_param[user, item] *= (1 - boost * 0.5)