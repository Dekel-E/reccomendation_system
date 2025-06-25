# Recommendation System Models: Baseline & SVD

This project contains Python implementations of two fundamental collaborative filtering algorithms for predicting movie ratings, developed as part of a university assignment on E-Commerce.

## Project Overview

The goal of this project is to predict user ratings for movies on a scale of 1 to 5. Two different models are implemented to achieve this:

1.  **Baseline Model:** A model based on user and item biases.
2.  **Matrix Factorization Model:** A model using Singular Value Decomposition (SVD) to find latent factors.

The performance of each model is evaluated using the Mean Squared Error (MSE) on the training data.

---

## Algorithms Implemented

### Task 1: Baseline Model with User/Item Biases (`task1.py`)

This model predicts a rating by starting with a global average and then adding learned biases for each user and item. The prediction formula is:

$$ \hat{r}_{u,i} = \mu + b_u + b_i $$

Where:
-   $\hat{r}_{u,i}$ is the predicted rating for user *u* on item *i*.
-   $\mu$ is the global average rating of all movies in the training set.
-   $b_u$ is the bias for user *u*, representing their tendency to rate higher or lower than average.
-   $b_i$ is the bias for item *i*, representing its tendency to receive higher or lower ratings than average.

The biases are found by solving a regularized least-squares problem to minimize the following objective function (with $\lambda=1$):

$$ L = \sum_{(u,i) \in \text{train}} (r_{u,i} - (\mu + b_u + b_i))^2 + \lambda(\sum_{u}b_u^2 + \sum_{i}b_i^2) $$

### Task 2: Singular Value Decomposition (SVD) (`task2.py`)

This model uses matrix factorization to find latent features that explain the observed ratings.

1.  **User-Item Matrix:** A sparse matrix is constructed from the training data, with users as rows, items as columns, and ratings as values. Missing ratings are filled with 0.
2.  **Low-Rank Approximation:** A sparse SVD is performed on the matrix to decompose it into a lower-dimensional space with a rank of **k=10**.
3.  **Prediction:** The original matrix is reconstructed from its low-rank components. This new, dense matrix contains the predicted ratings for all user-item pairs.

---

## File Structure

```
.
├── train.csv           # Training dataset with user ratings
├── test.csv            # Test dataset with user-item pairs for prediction
├── task1.py            # Script for the Baseline Model
├── task2.py            # Script for the SVD Model
├── pred1.csv           # (Generated) Predictions from task1.py
├── pred2.csv           # (Generated) Predictions from task2.py
└── me.txt              # (Generated) Final MSE scores for both tasks
```

---

## Requirements

The project requires the following Python libraries:

-   `pandas`
-   `numpy`
-   `scipy`

You can install them using pip:
```bash
pip install pandas numpy scipy
```

---

## How to Run

To run the models and generate the output files, follow these steps in order.

**1. Run the Baseline Model (Task 1):**

Execute the `task1.py` script from your terminal. Make sure `train.csv` and `test.csv` are in the same directory.

```bash
python task1.py
```

This will generate two files:
-   `pred1.csv`: The rating predictions for the test set.
-   `mse.txt`: A temporary file containing the MSE of the baseline model.

**2. Run the SVD Model (Task 2):**

Next, execute the `task2.py` script.

```bash
python task2.py
```

This script will:
-   Generate `pred2.csv` with the SVD model's predictions.
-   Read the MSE from `mse.txt`, append its own MSE on a new line, and rename the file to `me.txt`, which is the final submission file for the error scores.

After running both scripts, your directory will contain all the required output files.

##Todo part 2 - later
