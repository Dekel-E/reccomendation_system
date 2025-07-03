"""
Budget-Constrained Multi-Armed Bandit Recommender System

This module implements a recommendation system using Thompson Sampling, a Bayesian approach
to the multi-armed bandit problem. The system recommends items to users while respecting
a budget constraint.

Key Concepts:
- Multi-Armed Bandit (MAB): A problem where you must choose between multiple options (arms)
  to maximize reward, balancing exploration (trying new things) vs exploitation (using known good options)
- Thompson Sampling: A Bayesian approach that samples from posterior distributions to make decisions
- Beta Distribution: Used to model the probability of success for each user-item pair
- Budget Constraint: The total cost of recommended items cannot exceed a given budget

Educational Notes:
- Alpha (α): Represents successful interactions (prior successes + observed successes)
- Beta (β): Represents failed interactions (prior failures + observed failures)
- Higher α relative to β indicates higher success probability
- The Beta distribution is conjugate prior for Bernoulli outcomes (success/failure)
"""

import numpy as np
from itertools import combinations


class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        """
        Initialize the recommender system.
        
        Args:
            n_weeks (int): Number of recommendation rounds (weeks) to run
            n_users (int): Number of users to make recommendations for
            prices (np.array): Array of item prices
            budget (int): Total budget constraint for each round
            
        The system adapts its exploration strategy based on how tight the budget is:
        - Very tight budget (≤2 items affordable): Aggressive exploitation
        - Tight budget (≤4 items affordable): Balanced approach  
        - Normal budget (>4 items affordable): Standard exploration
        
        Thompson Sampling uses Beta distributions with parameters α (alpha) and β (beta):
        - α starts as prior belief in successes
        - β starts as prior belief in failures
        - Both get updated with observed outcomes
        """
        # Basic parameters
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget
        self.n_items = len(prices)
        
        # Budget analysis - determines exploration strategy
        self.min_price = np.min(prices)
        self.max_affordable = self.budget // self.min_price  # Maximum items we could possibly afford
        
        # Thompson Sampling parameters - adaptive based on budget constraints
        # The tighter the budget, the more we focus on exploitation vs exploration
        if self.max_affordable <= 2:
            # Very tight budget - aggressive exploitation
            # Start with uniform priors (α=1, β=1) and decay exploration quickly
            self.alpha = np.ones((n_users, self.n_items)) * 1
            self.beta_param = np.ones((n_users, self.n_items)) * 1
            self.exploration_decay = 30  # Stop exploring after 30 rounds
        elif self.max_affordable <= 4:
            # Tight budget - balanced approach
            # Higher α (2.5) vs β (0.4) = optimistic priors (assume items are good)
            self.alpha = np.ones((n_users, self.n_items)) * 2.5
            self.beta_param = np.ones((n_users, self.n_items)) * 0.4
            self.exploration_decay = 50
        else:
            # Normal budget - standard exploration
            # Balanced priors with moderate exploration
            self.alpha = np.ones((n_users, self.n_items)) * 2.0
            self.beta_param = np.ones((n_users, self.n_items)) * 0.5
            self.exploration_decay = 80
        
        # Statistics tracking - these track actual observed outcomes
        # Shape: (n_users, n_items) - tracks performance for each user-item pair
        self.trials = np.zeros((n_users, self.n_items))     # How many times we tried each user-item pair
        self.successes = np.zeros((n_users, self.n_items))  # How many times it was successful
        
        # Global popularity tracking - aggregated across all users
        # Shape: (n_items,) - overall item performance regardless of user
        self.global_trials = np.zeros(self.n_items)    # Total trials per item
        self.global_successes = np.zeros(self.n_items) # Total successes per item
        
        # Algorithm state
        self.round = 0                    # Current round number
        self.last_recommendations = None  # Last recommendations made (needed for updating)
        
        # Precompute all affordable item combinations to avoid recalculating each round
        # This is a key optimization for budget-constrained problems
        self._precompute_affordable_sets()
        
    def _precompute_affordable_sets(self):
        """
        Precompute all combinations of items that fit within the budget.
        
        This is a crucial optimization that solves the "knapsack-like" problem upfront.
        Instead of checking budget constraints during each recommendation, we precompute
        all valid item combinations and store them.
        
        Strategy depends on problem size:
        - Small problems (≤15 items): Enumerate all possible combinations
        - Large problems (>15 items): Use heuristics to sample good combinations
        
        Educational Note: This addresses the combinatorial explosion problem.
        With n items, there are 2^n possible combinations. We need smart strategies
        to avoid checking all of them.
        """
        self.affordable_sets = []
        
        if self.n_items <= 15:
            # Small problem: we can afford to check all combinations
            self._enumerate_all_affordable_sets()
        else:
            # Large problem: use heuristics to generate good combinations
            self._generate_sets_smartly()
    
    def _enumerate_all_affordable_sets(self):
        """
        Enumerate all affordable combinations for small problems.
        
        Uses the itertools.combinations function to generate all possible subsets
        of items and checks which ones fit within budget.
        
        Optimizations:
        - Limits maximum set size to avoid checking impossibly large combinations
        - Early termination if we find "enough" combinations (500)
        - Uses efficient numpy operations for price checking
        """
        # Practical limit: don't check combinations larger than what we can afford + buffer
        max_set_size = min(self.n_items, self.max_affordable + 2)
        
        # Check all combinations of increasing size
        for size in range(1, max_set_size + 1):
            # Stop if we already have enough combinations (computational efficiency)
            if len(self.affordable_sets) > 500:
                break
                
            # Generate all combinations of 'size' items
            for combo in combinations(range(self.n_items), size):
                # Check if this combination fits within budget
                if sum(self.item_prices[list(combo)]) <= self.budget:
                    self.affordable_sets.append(np.array(combo))
    
    def _generate_sets_smartly(self):
        """
        Generate item combinations using heuristics for large problems.
        
        When there are too many items to enumerate all combinations, we use
        smart strategies to generate a diverse set of good combinations:
        
        1. Include all affordable single items (baseline options)
        2. Prioritize cheaper items (more likely to fit in budget)
        3. Add some random combinations for diversity
        
        This is a classic example of using heuristics to make an intractable
        problem tractable while maintaining solution quality.
        """
        # Strategy 1: Always include all single items that fit budget
        # This ensures we have baseline options
        for i in range(self.n_items):
            if self.item_prices[i] <= self.budget:
                self.affordable_sets.append(np.array([i]))
        
        # Strategy 2: Focus on cheaper items (greedy approach)
        # Sort items by price to prioritize cheaper combinations
        sorted_items = np.argsort(self.item_prices)
        
        # Generate combinations starting from cheapest items
        for size in range(2, min(self.max_affordable + 1, 8)):  # Limit size for efficiency
            added = 0
            # Only consider cheapest items to reduce search space
            for combo in combinations(sorted_items[:min(self.n_items, 12)], size):
                if sum(self.item_prices[list(combo)]) <= self.budget:
                    self.affordable_sets.append(np.array(combo))
                    added += 1
                    if added > 100:  # Limit combinations per size
                        break
        
        # Strategy 3: Add random combinations for diversity
        # This helps avoid local optima by exploring unexpected combinations
        for _ in range(min(50, 500 - len(self.affordable_sets))):
            # Random set size
            size = np.random.randint(2, min(self.max_affordable + 1, 6))
            # Random item selection
            items = np.random.choice(self.n_items, size=size, replace=False)
            
            # Check budget and uniqueness
            if sum(self.item_prices[items]) <= self.budget:
                if not any(np.array_equal(items, s) for s in self.affordable_sets):
                    self.affordable_sets.append(items)
    
    def recommend(self) -> np.array:
        """
        Generate recommendations for all users using Thompson Sampling.
        
        This is the heart of the algorithm. The process involves:
        1. Sample from Beta distributions to get probability estimates
        2. Add exploration bonuses to encourage trying new items
        3. Find the best assignment of items to users within budget
        4. Return the recommended items for each user
        
        Returns:
            np.array: Array of item indices, one for each user
            
        Thompson Sampling Intuition:
        - For each user-item pair, we maintain a Beta(α, β) distribution
        - We sample from this distribution to get a "guess" at the success probability  
        - Items with higher sampled probabilities are more likely to be chosen
        - The randomness provides natural exploration
        """
        # Step 1: Thompson Sampling - sample probability estimates
        # Each entry is a random sample from Beta(α, β) for that user-item pair
        sampled_probs = np.random.beta(self.alpha, self.beta_param)
        
        # Step 2: Add exploration bonuses (only during exploration phase)
        if self.round < self.exploration_decay:
            # Variance-based exploration: items with high uncertainty get bonus
            # Formula from Beta distribution variance: αβ/((α+β)²(α+β+1))
            variance = (self.alpha * self.beta_param) / ((self.alpha + self.beta_param)**2 * 
                                                         (self.alpha + self.beta_param + 1))
            exploration_bonus = np.sqrt(variance) * (1 - self.round / self.exploration_decay) * 0.3
            
            # Bonus for completely untried items (encourage initial exploration)
            untried_bonus = (self.trials == 0) * 0.2 * (1 - self.round / self.exploration_decay)
            
            scores = sampled_probs + exploration_bonus + untried_bonus
        else:
            # Pure exploitation phase: just use sampled probabilities
            scores = sampled_probs
        
        # Step 3: Add global popularity bonus (wisdom of the crowd)
        if self.round > 20:  # Only after we have some data
            global_popularity = self.global_successes / (self.global_trials + 1)
            # Broadcasting: global_popularity[np.newaxis, :] makes it (1, n_items) 
            # so it can be added to scores which is (n_users, n_items)
            scores += 0.05 * global_popularity[np.newaxis, :]
        
        # Step 4: Find the best assignment of items to users within budget
        # This is the combinatorial optimization part
        best_value = -np.inf
        best_assignments = None
        
        # Try each precomputed affordable set of items
        for item_set in self.affordable_sets:
            # Get scores for this subset of items
            set_scores = scores[:, item_set]  # Shape: (n_users, len(item_set))
            
            # Assign items to users (two strategies based on set size)
            if len(item_set) >= self.n_users:
                # Case 1: More items than users - try to give each user a unique item
                # This maximizes diversity and avoids conflicts
                used = set()  # Track which items we've assigned
                assignments = np.zeros(self.n_users, dtype=int)
                
                for user in range(self.n_users):
                    best_score = -np.inf
                    best_item_idx = 0
                    
                    # Find best available item for this user
                    for idx, item in enumerate(item_set):
                        if idx not in used and set_scores[user, idx] > best_score:
                            best_score = set_scores[user, idx]
                            best_item_idx = idx
                    
                    used.add(best_item_idx)
                    assignments[user] = item_set[best_item_idx]
            else:
                # Case 2: Fewer items than users - assign greedily
                # Multiple users might get the same item
                best_items_idx = np.argmax(set_scores, axis=1)  # Best item for each user
                assignments = item_set[best_items_idx]
            
            # Calculate total expected value for this assignment
            expected_value = np.sum(scores[np.arange(self.n_users), assignments])
            
            # Small bonus for larger sets (more flexibility is valuable)
            expected_value += len(item_set) * 0.001
            
            # Keep track of the best assignment found so far
            if expected_value > best_value:
                best_value = expected_value
                best_assignments = assignments
        
        # Update state and return recommendations
        self.round += 1
        self.last_recommendations = best_assignments
        
        return best_assignments
    
    def update(self, results: np.array):
        """
        Update the recommendation system based on observed results.
        
        This is where the "learning" happens in our multi-armed bandit.
        We update our beliefs (Beta distribution parameters) based on
        whether the recommendations were successful or not.
        
        Args:
            results (np.array): Binary array indicating success (1) or failure (0)
                               for each user's recommended item
                               
        Bayesian Update Process:
        - Success: α += 1 (more evidence for success)
        - Failure: β += 1 (more evidence for failure)
        
        This is the beauty of Bayesian methods: new evidence naturally
        updates our uncertainty in a principled way.
        """
        if self.last_recommendations is None:
            return  # No recommendations to update
        
        # Get indices for the matrix updates
        user_indices = np.arange(self.n_users)  # [0, 1, 2, ..., n_users-1]
        item_indices = self.last_recommendations  # Items we recommended
        
        # Update per-user statistics
        # trials[user, item] counts how many times we tried this user-item pair
        # successes[user, item] counts how many times it worked
        self.trials[user_indices, item_indices] += 1
        self.successes[user_indices, item_indices] += results
        
        # Update Beta distribution parameters (Bayesian update)
        # α (alpha) represents total successes (prior + observed)
        # β (beta) represents total failures (prior + observed)
        self.alpha[user_indices, item_indices] += results
        self.beta_param[user_indices, item_indices] += (1 - results)  # failures
        
        # Update global statistics (aggregated across all users)
        for item in np.unique(item_indices):
            mask = item_indices == item  # Which users got this item
            self.global_trials[item] += np.sum(mask)
            self.global_successes[item] += np.sum(results[mask])
        
        # Optional: Confidence boosting for tight budgets
        # When budget is very tight, we boost confidence in consistently good items
        if self.max_affordable <= 3 and self.round > 50 and self.round % 25 == 0:
            # Calculate success rates (with safe division to avoid divide by zero)
            success_rates = np.divide(self.successes, self.trials,
                                    out=np.zeros_like(self.successes),
                                    where=self.trials > 10)
            
            # Identify consistently good items (>70% success rate, tried >10 times)
            good_items = (success_rates > 0.7) & (self.trials > 10)
            
            # Boost confidence: increase α (successes), decrease β (failures)
            # This makes the algorithm more confident about these good items
            self.alpha[good_items] *= 1.1
            self.beta_param[good_items] *= 0.9