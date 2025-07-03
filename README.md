# Recommendation System Models: Complete Implementation

![Python Badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Bandit Algorithm Badge](https://img.shields.io/badge/Bandit_Algorithm-D16A2E?style=for-the-badge&logo=git&logoColor=white)![SciPy Badge](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)![NumPy Badge](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)






This project contains Python implementations of multiple recommendation system approaches, developed as part of a university assignment on E-Commerce. The project is divided into two main parts, each exploring different aspects of recommendation systems.
Future techion students - Enjoy , dont blindcopy this please

## Project Overview

This comprehensive recommendation system project implements three different approaches:

**Part 1: Collaborative Filtering (Bias-based Recommendation System)**
1. **Baseline Model:** A model based on user and item biases
2. **Matrix Factorization Model:** A model using Singular Value Decomposition (SVD) to find latent factors

**Part 2: Multi-Armed Bandit Recommendation System**
3. **Stochastic MAB System:** A budget-constrained recommendation system using Thompson Sampling for exploration vs exploitation



---

## Algorithms Implemented

## Part 1: Collaborative Filtering Models (Bias-based Recommendation System)

### Task 1: Baseline Model with User/Item Biases (`biases_recsys/task1.py`)

This model predicts a rating by starting with a global average and then adding learned biases for each user and item. The prediction formula is:

$$ \hat{r}_{u,i} = \mu + b_u + b_i $$

Where:
-   $\hat{r}_{u,i}$ is the predicted rating for user *u* on item *i*.
-   $\mu$ is the global average rating of all movies in the training set.
-   $b_u$ is the bias for user *u*, representing their tendency to rate higher or lower than average.
-   $b_i$ is the bias for item *i*, representing its tendency to receive higher or lower ratings than average.

The biases are found by solving a regularized least-squares problem to minimize the following objective function (with $\lambda=1$):

$$ L = \sum_{(u,i) \in \text{train}} (r_{u,i} - (\mu + b_u + b_i))^2 + \lambda(\sum_{u}b_u^2 + \sum_{i}b_i^2) $$

### Task 2: Singular Value Decomposition (SVD) (`biases_recsys/task2.py`)

This model uses matrix factorization to find latent features that explain the observed ratings.

1.  **User-Item Matrix:** A sparse matrix is constructed from the training data, with users as rows, items as columns, and ratings as values. Missing ratings are filled with 0.
2.  **Low-Rank Approximation:** A sparse SVD is performed on the matrix to decompose it into a lower-dimensional space with a rank of **k=10**.
3.  **Prediction:** The original matrix is reconstructed from its low-rank components. This new, dense matrix contains the predicted ratings for all user-item pairs.

## Part 2: Multi-Armed Bandit Recommendation System

### Stochastic MAB with Budget Constraints (`stochastic_mab_recsys/`)

This advanced recommendation system tackles the **exploration vs exploitation** problem using **Thompson Sampling**, a Bayesian approach to multi-armed bandits. The system recommends items to users while respecting budget constraints.

#### Key Features:

**Thompson Sampling with Beta Distributions:**
- Maintains Beta(α, β) distributions for each user-item pair
- Naturally balances exploration (trying new items) vs exploitation (using known good items)
- Updates beliefs using Bayesian inference based on success/failure feedback

**Adaptive Exploration Strategy:**
- Adjusts exploration intensity based on budget tightness
- Tight budgets → more exploitation (less risky)
- Loose budgets → more exploration (can afford mistakes)

**Budget-Constrained Optimization:**
- Precomputes all affordable item combinations
- Uses smart heuristics for large item catalogs
- Assigns items to users optimally within budget constraints

**Key Components:**
- `Recommender.py`: Main Thompson Sampling implementation
- `simulation.py`: Framework for testing and comparing algorithms
- `run_tests.py`: Automated testing and evaluation

#### Algorithm Flow:
1. **Initialization**: Set up Beta priors based on budget analysis
2. **Recommendation**: Sample from Beta distributions and solve assignment problem
3. **Feedback**: Observe success/failure results
4. **Update**: Bayesian update of Beta parameters
5. **Repeat**: Continue learning and improving over time

This approach is particularly valuable for:
- E-commerce platforms with marketing budgets
- Content recommendation with bandwidth constraints  
- Resource allocation problems
- Any scenario requiring online learning with constraints

---

## File Structure

```
.
├── biases_recsys/              # Part 1: Collaborative Filtering Models
│   ├── train.csv              # Training dataset with user ratings
│   ├── test.csv               # Test dataset for prediction
│   ├── task1.py               # Baseline Model implementation
│   ├── task2.py               # SVD Model implementation
│
├── stochastic_mab_recsys/      # Part 2: Multi-Armed Bandit System
│   ├── Recommender.py         # Main Thompson Sampling implementation
│   ├── simulation.py          # Testing framework
│   ├── run_tests.py           # Automated evaluation
│   ├── test.py                # Unit tests
│   ├── test_2.py              # Additional testing
│
├── README.md                   # This file
```

---

## Requirements

The project requires the following Python libraries:

**For Part 1 (Collaborative Filtering):**
-   `pandas`
-   `numpy`
-   `scipy`

**For Part 2 (Multi-Armed Bandit):**
-   `numpy`
-   `itertools` (built-in)

You can install the required packages using pip:
```bash
pip install pandas numpy scipy
```

---

## How to Run

### Part 1: Collaborative Filtering Models

To run the bias-based recommendation models, navigate to the `biases_recsys/` directory and follow these steps:

**1. Run the Baseline Model (Task 1):**

```bash
cd biases_recsys
python task1.py
```

This will generate:
-   `pred1.csv`: Rating predictions for the test set
-   `mse.txt`: MSE of the baseline model

**2. Run the SVD Model (Task 2):**

```bash
python task2.py
```

This will generate:
-   `pred2.csv`: SVD model predictions
-   `me.txt`: Final MSE scores for both models

### Part 2: Multi-Armed Bandit System

To run the stochastic MAB recommendation system, navigate to the `stochastic_mab_recsys/` directory:

**1. Run Individual Tests:**

```bash
cd stochastic_mab_recsys
python test.py
```

**2. Run Comprehensive Evaluation:**

```bash
python run_tests.py
```

**3. Run Custom Simulation:**

```bash
python simulation.py
```

**Key Files to Examine:**
- `Recommender.py`: The main Thompson Sampling implementation with detailed documentation
- `ALGORITHM_EXPLANATION.md`: Comprehensive guide to understanding the algorithm
- `simulation.py`: Framework for testing different scenarios

## Educational Value

This project demonstrates:

**Part 1 - Collaborative Filtering:**
- Matrix factorization techniques
- Bias modeling in recommendation systems
- Regularized optimization
- SVD for dimensionality reduction

**Part 2 - Multi-Armed Bandit:**
- Bayesian inference and Thompson Sampling
- Exploration vs exploitation trade-offs
- Budget-constrained optimization
- Online learning algorithms
- Combinatorial optimization under constraints

Both parts showcase different paradigms in recommendation systems - from traditional collaborative filtering to modern online learning approaches.
