import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csc_matrix, identity
from scipy.sparse.linalg import spsolve


def solve():
    # Load the training data
    train_df = pd.read_csv('train.csv')
    # Load the test data
    test_df = pd.read_csv('test.csv')

    #calculate global mean
    r_avg=train_df['rating'].mean()

    # Get unique users and items
    unique_users = sorted(pd.concat([train_df['user id'], test_df['user id']]).unique())
    unique_items = sorted(pd.concat([train_df['item id'], test_df['item id']]).unique())
    
    #Get number of unique users and items
    n_users = len(unique_users)
    n_items = len(unique_items)
    n_ratings = len(train_df)
    #Create mappings from user/item IDs to indices
    user_to_index = {user_id: i for i, user_id in enumerate(unique_users)}
    item_to_index = {item_id: i for i, item_id in enumerate(unique_items)}
    print(f"Found {n_users} unique users and {n_items} unique items.")

    #Build and solve the linear system
    # Create a sparse matrix for the ratings
    print("Creating sparse matrix...")
    A= lil_matrix((n_ratings,n_users+n_items), dtype=np.float32) #Create n_ratings x (n_users + n_items) sparse matrix
    c= np.zeros((n_ratings,1), dtype=np.float32) #Create a vector for the ratings 

    #Fill matrix row by row
    for i,row in train_df.iterrows():
        user_index = user_to_index[row['user id']]
        item_index = item_to_index[row['item id']] 
        A[i, user_index] = 1
        A[i, n_users+item_index] = 1
        c[i] = row['rating'] - r_avg

    # Convert to CSC(compressed Sparse Column) format for efficient arithmetic operations
    A=A.tocsc()
    #Regularization parameter
    lambda_val=1

    #what we need to solve is A.T @ A + lambda * I
    AtA =A.T @ A #A transpose times A
    I = identity(n_users+n_items, format='csc', dtype=np.float32)
    left_side = AtA + lambda_val * I #(A^T A + lambda * I)
    right_side = A.T @ c #A^T c

    #Solve using sparse linear solver(spsolve)
    b = spsolve(left_side, right_side)
    #split vector to user and item biases
    user_biases = b[:n_users]
    item_biases = b[n_users:]
    print("Biases calculated.")
    user_biases  = {uid:bias for uid, bias in zip(unique_users, user_biases)}
    item_biases  = {iid:bias for iid, bias in zip(unique_items, item_biases)}

    #Calculate MSE on Training set and generate predictions
    print("Calculating MSE on training set...")
    train_predictions = []
    for i, row in train_df.iterrows():
        user_id = row['user id']
        item_id = row['item id']
        
        b_u = user_biases.get(user_id, 0)
        b_i = item_biases.get(item_id, 0)
        prediction = r_avg + b_u + b_i
        train_predictions.append(prediction)
    train_mse = np.mean((train_df['rating'] - train_predictions) ** 2)
    print(f"Training MSE: {train_mse:.4f}")

    #Genereate and save predictions for the test set
    print("Generating predictions for the test set...")
    test_predictions = []
    for i, row in test_df.iterrows():
        user_id = row['user id']
        item_id = row['item id']

        b_u = user_biases.get(user_id, 0)
        b_i = item_biases.get(item_id, 0)
        prediction = r_avg + b_u + b_i
        prediction = np.clip(prediction, 1, 5) # Clip predictions to the range [1, 5]
        test_predictions.append(prediction)

    submission_df = test_df.copy()
    submission_df['rating'] = test_predictions
    print("Saving predictions to 'pred1.csv'...")
    submission_df.to_csv('pred1.csv', index=False)
    #Save MSE to file
    try:
        with open('mse.txt', 'w') as f:
            f.write(str(train_mse))
    except Exception as e:
            print(f"Error saving MSE: {e}")
    


if __name__ == "__main__":
    solve()
