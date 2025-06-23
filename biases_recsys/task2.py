import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

def solve():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure 'train.csv' and 'test.csv' are in the same directory.")
        return
    
    # Get all unique users and items from both train and test sets to define the matrix dimensions
    all_users = sorted(pd.concat([train_df['user id'], test_df['user id']]).unique())
    all_items = sorted(pd.concat([train_df['item id'], test_df['item id']]).unique())

    n_users = len(all_users)
    n_items = len(all_items)
     # Create mappings from original IDs to zero-based indices for the matrix
    user_to_index = {user_id: i for i, user_id in enumerate(all_users)}
    item_to_index = {item_id: i for i, item_id in enumerate(all_items)}

    rows = train_df['user id'].map(user_to_index).values
    cols = train_df['item id'].map(item_to_index).values
    ratings = train_df['rating'].values

    # Create a sparse matrix in CSC format
    ratings_matrix = csc_matrix((ratings, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    print("performing SVD...")
    k=10
    U , s , Vt =svds(ratings_matrix, k=k)

    #function aboce returns singular values in a 1D array, we need to convert it to a diagonal matrix
    s_diag = np.diag(s)
    # Reconstruct the ratings matrix
    predicted_ratings = np.dot(np.dot(U, s_diag), Vt)

    print("Calculageting MSE...")
    squared_errors = []
    for i, row in train_df.iterrows():
        user_idx = user_to_index[row['user id']]
        item_idx = item_to_index[row['item id']]
        actual_rating = row['rating']
        predicted_rating = predicted_ratings[user_idx, item_idx]
        squared_errors.append((predicted_rating-actual_rating) ** 2)

    mse = np.mean(squared_errors)
    print(f"MSE: {mse}")

    #Generate predictions for the test set
    print("Generating predictions for the test set...")
    test_predictions = []
    for i, row in test_df.iterrows():
        user_idx = user_to_index.get(row['user id'], None)
        item_idx = item_to_index.get(row['item id'], None)

        if user_idx is not None and item_idx is not None:
            pred = predicted_ratings[user_idx, item_idx]
            # As per instructions, clip predictions to be between 1 and 5
            pred = np.clip(pred, 1, 5)
            test_predictions.append(pred)
        else:
            # Fallback for safety, e.g., predict a global average if needed
            test_predictions.append(train_df['rating'].mean())

    #Create submission DataFrame
    submission_df = test_df.copy()
    submission_df['rating'] = test_predictions
    submission_df.to_csv('pred2.csv', index=False)
    
    with open('mse.txt', 'r') as f:
        mse1 = f.read().strip()
        
    with open('mse.txt', 'w') as f:
        f.write(f"{mse1}\n")  # Write MSE from Task 1 (with a newline)
        f.write(str(mse))     # Write MSE from Task 2 on the second line

if __name__ == '__main__':
    solve()