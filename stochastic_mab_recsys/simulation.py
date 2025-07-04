import time
from test import test_1,test_2,test_3

from Recommender import Recommender
import numpy as np


TOTAL_TIME_LIMIT = 120 # seconds

class Simulation():
    def __init__(self, P: np.array, prices: np.array, budget, n_weeks: int):
        self.P = P.copy()
        self.item_prices = prices
        self.budget = budget
        self.n_weeks = n_weeks

    def _validate_recommendation(self, recommendation):
        if not isinstance(recommendation, np.ndarray):
            print(f'ERROR: {recommendation} is not an np.array')
            return False
        
        if not np.issubdtype(recommendation.dtype, np.integer):
            print(f'ERROR: type of {recommendation} is not int')
            return False
        
        if recommendation.shape != (self.P.shape[0],):
            print(f'ERROR: {recommendation} is not 1D array or has wrong length')
            return False
        
        if ((recommendation < 0) | (recommendation >= self.P.shape[1])).any():
            print(f'ERROR: {recommendation} contains invalid podcasts')
            return False

        podcasts = np.unique(recommendation)
        total_price = np.sum(self.item_prices[podcasts])
        
        if total_price > self.budget:
            print(f'ERROR: {total_price} is above budget of {self.budget}')
            return False
            
        return True
    
    def simulate(self) -> int:
        total_time_taken = 0
        
        init_start = time.perf_counter()
        
        try:
            recommender = Recommender(n_weeks=self.n_weeks, n_users=self.P.shape[0], 
                                      prices=self.item_prices.copy(), 
                                      budget=self.budget)
        except Exception as e:
            print('Recommender __init__ caused error')
            raise e
        
        init_end = time.perf_counter()
        
        total_time_taken += init_end - init_start
        
        reward = 0
              
        for round_idx in range(self.n_weeks):
            try:
                recommendation_start = time.perf_counter()
                recommendation = recommender.recommend()
                recommendation_end = time.perf_counter()
            except Exception as e:
                print(f'Recommmender.recommend() raised error at round {round_idx}')
                raise e 
                
            if recommendation is None:
                print('No recommendation supplied.')
                return 0
                
            recommendation_time = recommendation_end - recommendation_start
                
            if not self._validate_recommendation(recommendation):
                print(f'Error: Invalid recommendation at round {round_idx}')
                return 0
            
            results = np.random.binomial(n=1, p=self.P[np.arange(self.P.shape[0]), recommendation])
            current_reward = np.sum(results)
            next_reward = reward + current_reward
            
            try:
                update_start_time = time.perf_counter()
                recommender.update(results)
                update_end_time = time.perf_counter()
            except Exception as e:
                print(f'Recommmender.update() raised error at round {round_idx}')
                raise e
            
            update_time = update_end_time - update_start_time

            time_for_current_round = recommendation_time + update_time

            if total_time_taken + time_for_current_round > TOTAL_TIME_LIMIT:
                print(f'TOTAL TIME LIMIT EXCEEDED. Returning reward at after {round_idx} rounds')
                return reward
            else:
                total_time_taken += time_for_current_round
                reward = next_reward
        
        print(f'Total time taken: {total_time_taken} seconds')
        return reward
    
if __name__ == '__main__':
    test_1_res = []
    test_2_res = []
    test_3_res = []
    for i in range(30):
        simulation = Simulation(test_1['P'], test_1['item_prices'], test_1['budget'], test_1['n_weeks'])
        reward = simulation.simulate()
        test_1_res.append(reward)
        print(f'Test 1 trial {i+1} reward: {reward}')
        
        # simulation = Simulation(test_2['P'], test_2['item_prices'], test_2['budget'], test_2['n_weeks'])
        # reward = simulation.simulate()
        # test_2_res.append(reward)
        # print(f'Test 2 trial {i+1} reward: {reward}')
        # simulation = Simulation(test_3['P'], test_3['item_prices'], test_3['budget'], test_3['n_weeks'])
        # reward = simulation.simulate()
        # test_3_res.append(reward)
        # print(f'Test 3 trial {i+1} reward: {reward}')

       
    
    print(f'Test 1 results: mean: {np.mean(test_1_res)}, std: {np.std(test_1_res)}') 
    # print(f'Test 2 results mean: {np.mean(test_2_res)} std: {np.std(test_2_res)}')  
    # print(f'Test 3 results mean: {np.mean(test_3_res)} std: {np.std(test_3_res)}')