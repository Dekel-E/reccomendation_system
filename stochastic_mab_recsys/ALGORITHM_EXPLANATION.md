# Budget-Constrained Multi-Armed Bandit Recommender System

## Overview

This code implements a sophisticated recommendation system that solves the **Multi-Armed Bandit (MAB)** problem with budget constraints using **Thompson Sampling**. It's designed for scenarios where you need to recommend items to users while staying within a budget.

## Key Concepts

### 1. Multi-Armed Bandit Problem
- **Analogy**: Imagine a casino with multiple slot machines (arms), each with unknown payout rates
- **Challenge**: Balance **exploration** (trying new machines) vs **exploitation** (using known good machines)
- **Goal**: Maximize total reward over time

### 2. Thompson Sampling
- **Type**: Bayesian approach to MAB
- **Core Idea**: Maintain probability distributions for each arm's success rate
- **Decision Making**: Sample from these distributions and choose the arm with highest sampled value
- **Advantage**: Natural exploration through randomness in sampling

### 3. Beta Distribution
- **Purpose**: Models the probability of success for each user-item pair
- **Parameters**: 
  - α (alpha): Successes + prior belief in successes
  - β (beta): Failures + prior belief in failures
- **Properties**: 
  - Mean ≈ α/(α+β)
  - Higher α relative to β = higher success probability
  - **Conjugate prior** for Bernoulli outcomes (perfect for success/failure)

## Algorithm Flow

### Initialization
1. **Budget Analysis**: Determine exploration strategy based on budget tightness
2. **Prior Setting**: Initialize Beta distributions with strategic priors
3. **Affordable Sets**: Precompute all item combinations that fit within budget

### Recommendation Process (each round)
1. **Thompson Sampling**: Sample success probabilities from Beta distributions
2. **Exploration Bonuses**: Add bonuses for uncertain or untried items
3. **Combinatorial Optimization**: Find best assignment of items to users within budget
4. **Return Recommendations**: Output one item per user

### Learning Process (after each round)
1. **Observe Results**: Get binary feedback (success/failure) for each recommendation
2. **Bayesian Update**: Update Beta parameters based on observed outcomes
3. **Statistics Tracking**: Update both individual and global statistics

## Key Algorithmic Innovations

### 1. Adaptive Exploration Strategy
```python
if self.max_affordable <= 2:
    # Very tight budget - aggressive exploitation
    self.exploration_decay = 30
elif self.max_affordable <= 4:
    # Tight budget - balanced approach  
    self.exploration_decay = 50
else:
    # Normal budget - standard exploration
    self.exploration_decay = 80
```

**Why?** When budget is tight, mistakes are costly. The algorithm adapts by:
- Reducing exploration time
- Using more aggressive priors
- Focusing on exploitation sooner

### 2. Smart Affordable Set Generation
For large problems, instead of checking all 2^n combinations:
- Include all single affordable items
- Prioritize cheaper items (greedy approach)
- Add random combinations for diversity

**Complexity**: Reduces from O(2^n) to O(n²) while maintaining solution quality

### 3. Multi-Level Exploration Bonuses
The algorithm uses multiple exploration mechanisms:
- **Variance bonus**: Items with high uncertainty get exploration bonus
- **Untried bonus**: Never-tried items get extra encouragement
- **Global popularity**: Wisdom of the crowd effect

### 4. Intelligent Assignment Strategy
When assigning items to users:
- **More items than users**: Give each user a unique item (maximize diversity)
- **Fewer items than users**: Use greedy assignment (some users get same item)

## Educational Value

### Bayesian Learning
This is a perfect example of **Bayesian learning**:
- Start with prior beliefs (initial α, β values)
- Update beliefs with evidence (observed outcomes)
- Uncertainty naturally decreases as we gather more data
- No need for complex learning rates or convergence criteria

### Exploration vs Exploitation
The algorithm beautifully demonstrates this fundamental trade-off:
- **Early rounds**: High exploration (try new things)
- **Later rounds**: More exploitation (use what we know works)
- **Budget constraints**: Force earlier shift to exploitation

### Combinatorial Optimization
Shows how to handle complex constraints:
- Budget constraint creates combinatorial problem
- Smart preprocessing (affordable sets) makes it tractable
- Heuristics balance optimality with computational efficiency

## Real-World Applications

1. **E-commerce**: Recommend products within marketing budget
2. **Content Platforms**: Suggest content with bandwidth constraints
3. **Advertising**: Allocate ad spend across different campaigns
4. **Resource Allocation**: Distribute limited resources among users

## Mathematical Foundation

### Beta Distribution Update
When we observe result r ∈ {0,1}:
- α_new = α_old + r
- β_new = β_old + (1-r)

This is the **conjugate prior** property - Beta is conjugate to Bernoulli.

### Thompson Sampling
For each arm i:
1. Sample θ_i ~ Beta(α_i, β_i)
2. Choose arm with highest θ_i

### Exploration Decay
Exploration bonus = f(variance, round) where f decreases over time
- Early: High uncertainty → high bonus → more exploration
- Later: Low uncertainty → low bonus → more exploitation

## Why This Approach Works

1. **Principled Uncertainty**: Beta distributions naturally represent our uncertainty
2. **Automatic Balancing**: Thompson Sampling inherently balances exploration/exploitation
3. **Efficient Computation**: Smart preprocessing makes real-time decisions fast
4. **Adaptive Strategy**: Algorithm adapts to budget constraints automatically
5. **Robust Performance**: Works well across different problem sizes and constraints

This implementation showcases advanced concepts in:
- Bayesian statistics
- Online learning
- Combinatorial optimization
- Algorithm design for constrained problems
