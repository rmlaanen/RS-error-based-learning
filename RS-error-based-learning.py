# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:13:36 2023

@author: rmlaanen
"""

# In[1]:

# PREPARING ENVIRONMENT    

from __future__ import division
import pandas as pd
import math as math
import random
import numpy as np
import collections
import os
import warnings
from surprise.model_selection import RandomizedSearchCV
from surprise import SVD, Reader, accuracy, Dataset

# In[2]:
    
# LOADING AND TRANSFORMING DATASET    

# Set working directory
os.chdir("...")

# Load dataset
data_pd = pd.read_csv("useritemmatrix.csv")

# Rename columns to fit surprise package
data_pd.rename(columns={'userId': 'user_id', 'itemId': 'item_id', 'interaction': 'raw_ratings'}, inplace=True)

# Set working directory
os.chdir("...")

print('Loaded dataset')

# In[3]:
    
# REDUCE DATASET SIZE

# Count the number of interactions for each user and item
user_counts = data_pd['user_id'].value_counts()
item_counts = data_pd['item_id'].value_counts()

# Filter user_ids with at least 50 interactions
valid_user_ids = user_counts[user_counts >= 0].index

# Filter item_ids with at least 50 interactions
valid_item_ids = item_counts[item_counts >= 0].index

# Filter the dataset to include interactions with both valid user_ids and valid item_ids
data_pd = data_pd[data_pd['user_id'].isin(valid_user_ids) & data_pd['item_id'].isin(valid_item_ids)]

# Reduce dataset size to 100.000 interactions
sample_size = 100000
data_pd = data_pd.sample(n=sample_size, random_state=123)

# Transform data for surprise package
data = Dataset.load_from_df(data_pd[['user_id', 'item_id', 'raw_ratings']],
                            reader = Reader(rating_scale=(0, 1))) 

print('Prepared dataset')

# In[4]:
    
# HYPERPARAMETER TUNING

# Record optimal hyperparameters
factors = 100
reg_b = 1e-08
reg_q = 1e-05

print('Performed hyperparameter tuning')
print('Optimal number of factors: ' + str(factors))
print('Optimal bias regularization strength: ' + str(reg_b))
print('Optimal latent factor regularization strength: ' + str(reg_q))

# In[6]:
    
# RANDOMLY SELECTING COLD USERS

# Order users by interaction amount
user_freq_df = pd.DataFrame.from_dict(collections.Counter(data_pd['user_id']),orient='index').reset_index()
user_freq_df = user_freq_df.rename(columns={'index':'user_id', 0:'freq'})

# Define percentage of users to be set as cold users
perc_cold_users = 0.25
nr_of_cold_users = int(math.floor(len(user_freq_df)*perc_cold_users))

# Select the [nr_of_cold_users] with the highest number of interactions
cold_users = user_freq_df.sort_values(by='freq',ascending=False).head(nr_of_cold_users)
cold_users = user_freq_df.sort_values(by='freq', ascending=False).head(nr_of_cold_users)
cold_users = cold_users.iloc[:, 0]

print('Selected ' + str(nr_of_cold_users) + ' cold users')

# In[7]:
    
# SETTINGS FOR SHOWN ITEMS

# Compute interaction frequency per item
item_freq_counter = collections.Counter(data_pd['item_id'])
item_freq_df = pd.DataFrame.from_dict(item_freq_counter,orient='index').reset_index()
item_freq_df = item_freq_df.rename(columns={'index':'item_id', 0:'freq'})

# Produce list of items with at least 10 interactions
threshold_item = 10
threshold_item_df = item_freq_df[item_freq_df['freq']>=threshold_item]['item_id']
threshold_freq_df = item_freq_df[item_freq_df['freq']>=threshold_item]
    
# create list of all unrated items meeting the threshold
candidate_item_list = item_freq_df[item_freq_df['freq']>=threshold_item]['item_id']
nr_of_candidate_items = int(math.floor(len(candidate_item_list)*perc_cold_users))

# create list of possible ratings
minimum_rating = 0
maximum_rating = 1
rating_list = list(range(minimum_rating,maximum_rating+1))

print('Selected ' + str(len(threshold_freq_df)) + ' candidate items')

# In[8]:
    
# CREATE DATASET WITHOUT COLD USER OBSERVATIONS TO PRODUCE ITEM RANKING

# Create a boolean mask to filter cold users
cold_users_mask = data_pd['user_id'].isin(cold_users)

# Use the mask to split the dataset
data_pd_cold = data_pd[cold_users_mask]
data_pd_warm = data_pd[~cold_users_mask]

# In[9]:
    
# CREATING A SIMPLE SVD MODEL
    
# Create the SVD model
model = SVD(n_factors=factors, 
                    n_epochs=100,
                    biased=True,
                    reg_all=None,
                    lr_bu=None,
                    lr_bi=None,
                    lr_pu=None,
                    lr_qi=None,
                    reg_bu=reg_b,
                    reg_bi=reg_b,
                    reg_pu=reg_q,
                    reg_qi=reg_q,
                    random_state=123,
                    verbose=False)

print('Created SVD model')

# In[10]:
    
### Y-CHANGE
    
# Split the dataset into training (70%) and test (30%)
split_ratio = 0.3
df_sampled = data_pd_warm.sample(frac=1, random_state=123) 
num_test_samples = int(len(data_pd_warm) * split_ratio)
y_change_train_data_pd = df_sampled.iloc[num_test_samples:]
y_change_test_data_pd = df_sampled.iloc[:num_test_samples]
    
# transform data for surprise package
y_change_train_data = Dataset.load_from_df(y_change_train_data_pd[['user_id', 'item_id', 'raw_ratings']],
                            reader = Reader(rating_scale=(0, 1))) 
y_change_train_data = y_change_train_data.build_full_trainset()
    
# Train the model on the original train set
model.fit(y_change_train_data)
    
# Predict test item ratings
y_change_test_data_pd.loc[:, 'predicted_rating'] = y_change_test_data_pd.apply(
    lambda row: model.predict(row['user_id'], row['item_id']).est, axis=1)

# Initialize a list to store candidate items and their associated total differences
y_change_results_risky = pd.DataFrame(columns=['item_id', 'Y-change'])
y_change_results_moderate = pd.DataFrame(columns=['item_id', 'Y-change'])
y_change_results_conservative = pd.DataFrame(columns=['item_id', 'Y-change'])

# Progress tracking
idx = 0
total_candidate_items = len(candidate_item_list)

# Disable Warnings
warnings.simplefilter("ignore", category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

# Iterate through each candidate training point i_x in set of interacted items

for i_x in candidate_item_list:
    
    # Add count for progress
    idx = idx + 1
    
    # Initialize variable to track the best rating
    y_change_candidate_risky = float('inf')
    y_change_candidate_moderate = 0
    y_change_candidate_conservative = 0
    
    # Iterate through each rating 'y' in 'Y' (0 and 1)
    for y in rating_list:
        
        # Create new training data by adding the candidate item with rating 'y'  
        y_change_new_train_data_pd = y_change_train_data_pd.append(pd.DataFrame({'user_id': [0], 'item_id': [i_x], 'raw_ratings': [y]}))
        y_change_new_train_data = Dataset.load_from_df(y_change_new_train_data_pd[['user_id', 'item_id', 'raw_ratings']],
                                    reader = Reader(rating_scale=(0, 1))) 
        y_change_new_train_data = y_change_new_train_data.build_full_trainset()
        
        # Train recommender system using the new training data
        model.fit(y_change_new_train_data)
        
        # Initialize total difference for this candidate training point and rating
        y_change_candidate = 0
        
        # Iterate through each item i in the test set (I_test) and predict the rating
        for i in y_change_test_data_pd['item_id'].unique():
            
            # Select test data for the current item i
            y_change_test_data_pd_item = y_change_test_data_pd[y_change_test_data_pd['item_id'] == 	i]
            
            # Predict user ratings for the test item using the new training data
            y_change_predictions = model.predict(y_change_test_data_pd_item.iloc[0]['user_id'], y_change_test_data_pd_item.iloc[0]['item_id']).est
            
            # Calculate the squared difference in rating estimates for this candidate and rating 'y' on the test item
            y_change_candidate += ((y_change_predictions - y_change_test_data_pd_item['predicted_rating'].values[0]) ** 2)
        
        # Select best / worst for risky and conservative strategies
        if y_change_candidate < y_change_candidate_risky:
            y_change_candidate_risky = y_change_candidate
        y_change_candidate_moderate += y_change_candidate      
        if y_change_candidate > y_change_candidate_conservative:
            y_change_candidate_conservative = y_change_candidate
            
    # Multiply by the normalizing constant
    y_change_candidate_moderate = (1/2)*y_change_candidate_moderate
        
    # Append the candidate item and its associated total difference to the results list
    data_to_append_risky = {'item_id': i_x, 'Y-change': y_change_candidate_risky}
    y_change_results_risky = y_change_results_risky.append(data_to_append_risky, ignore_index=True)
    data_to_append_moderate = {'item_id': i_x, 'Y-change': y_change_candidate_moderate}
    y_change_results_moderate = y_change_results_moderate.append(data_to_append_moderate, ignore_index=True)
    data_to_append_conservative = {'item_id': i_x, 'Y-change': y_change_candidate_conservative}
    y_change_results_conservative = y_change_results_conservative.append(data_to_append_conservative, ignore_index=True)
    print('Y-change: appended ' + 'Item ' + str(idx) + ' out of ' + str(total_candidate_items))

# Re-enable warnings
warnings.resetwarnings()
pd.set_option('mode.chained_assignment', 'warn')

# Store results
y_change_items_risky_df = y_change_results_risky
y_change_items_risky_df.sort_values(by='Y-change',inplace=True,ascending=False)
y_change_items_moderate_df = y_change_results_moderate
y_change_items_moderate_df.sort_values(by='Y-change',inplace=True,ascending=False)
y_change_items_conservative_df = y_change_results_conservative
y_change_items_conservative_df.sort_values(by='Y-change',inplace=True,ascending=False)

print('Computed Y-change scoring')

# In[11]:
    
### CV-BASED

# Split the dataset into training (70%) and test (30%)
split_ratio = 0.3
df_sampled = data_pd_warm.sample(frac=1, random_state=123) 
num_test_samples = int(len(data_pd_warm) * split_ratio)
cv_based_train_data_pd = df_sampled.iloc[num_test_samples:]
cv_based_test_data_pd = df_sampled.iloc[:num_test_samples]
    
# transform data for surprise package
cv_based_train_data = Dataset.load_from_df(cv_based_train_data_pd[['user_id', 'item_id', 'raw_ratings']],
                            reader = Reader(rating_scale=(0, 1))) 
cv_based_train_data = cv_based_train_data.build_full_trainset()

# Initialize a list to store candidate items and their associated total differences
cv_based_results_risky = pd.DataFrame(columns=['CandidateItem', 'CV_based'])
cv_based_results_moderate = pd.DataFrame(columns=['CandidateItem', 'CV_based'])
cv_based_results_conservative = pd.DataFrame(columns=['CandidateItem', 'CV_based'])

# Progress tracking
idx = 0
total_candidate_items = len(candidate_item_list)
warnings.simplefilter("ignore", category=FutureWarning)

# Iterate through each candidate training point i_x in set of interacted items Iu

for i_x in candidate_item_list:
    
    # Add count for progress
    idx = idx + 1
    
    # Initialize variable to track the best rating
    cv_based_candidate_risky = float('inf')
    cv_based_candidate_moderate = 0
    cv_based_candidate_conservative = 0
    
    # Iterate through each rating 'y' in 'Y' (0 and 1)
    for y in rating_list:
        
        # Create new training data by adding the candidate item with rating 'y'  
        cv_based_new_train_data_pd = cv_based_train_data_pd.append(pd.DataFrame({'user_id': [0], 'item_id': [i_x], 'raw_ratings': [y]}))
        cv_based_new_train_data = Dataset.load_from_df(cv_based_new_train_data_pd[['user_id', 'item_id', 'raw_ratings']],
                                    reader = Reader(rating_scale=(0, 1))) 
        cv_based_new_train_data = cv_based_new_train_data.build_full_trainset()
        
        # Train recommender system using the new training data
        model.fit(cv_based_new_train_data)
        
        # Initialize total squared difference in rating estimates for this candidate item i_x and rating y
        cv_based_candidate = 0
        
        # Iterate through each item i in the test set (I_test) and predict the rating
        for i in cv_based_test_data_pd['item_id'].unique():
            
            # Select test data for the current item i
            cv_based_test_data_pd_item = cv_based_test_data_pd[cv_based_test_data_pd['item_id'] == 	i]
            
            # Predict user ratings for the test item using the new training data
            cv_based_predictions = model.predict(cv_based_test_data_pd_item.iloc[0]['user_id'], cv_based_test_data_pd_item.iloc[0]['item_id']).est
            
            # Calculate the squared difference in rating estimates for this candidate and rating 'y' on the test item
            cv_based_candidate += ((cv_based_predictions - cv_based_test_data_pd_item['raw_ratings'].values[0]) ** 2)
        
        # Select best / worst for risky and conservative strategies
        if cv_based_candidate < cv_based_candidate_risky:
            cv_based_candidate_risky = cv_based_candidate
        cv_based_candidate_moderate += cv_based_candidate      
        if cv_based_candidate > cv_based_candidate_conservative:
            cv_based_candidate_conservative = cv_based_candidate
            
    # Multiply by the normalizing constant
    cv_based_candidate_moderate = (1/2)*y_change_candidate_moderate
        
    # Append the candidate item and its associated total difference to the results list
    data_to_append_risky = {'item_id': i_x, 'CV_based': cv_based_candidate_risky}
    cv_based_results_risky = cv_based_results_risky.append(data_to_append_risky, ignore_index=True)
    data_to_append_moderate = {'item_id': i_x, 'CV_based': cv_based_candidate_moderate}
    cv_based_results_moderate = cv_based_results_moderate.append(data_to_append_moderate, ignore_index=True)
    data_to_append_conservative = {'item_id': i_x, 'CV_based': cv_based_candidate_conservative}
    cv_based_results_conservative = cv_based_results_conservative.append(data_to_append_conservative, ignore_index=True)
    print('CV-based: appended ' + 'Item ' + str(idx) + ' out of ' + str(total_candidate_items))

# Store results
cv_based_items_risky_df = cv_based_results_risky
cv_based_items_risky_df.sort_values(by='CV_based',inplace=True,ascending=False)
cv_based_items_moderate_df = cv_based_results_moderate
cv_based_items_moderate_df.sort_values(by='CV_based',inplace=True,ascending=False)
cv_based_items_conservative_df = cv_based_results_conservative
cv_based_items_conservative_df.sort_values(by='CV_based',inplace=True,ascending=False)

warnings.resetwarnings()
print('Computed CV-based scoring')

# In[12]:
    
### POPGINI

# Define Gini score function
def gini(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    gini = 0.
    
    sum_probs = 0
    
    for iterator in probs:
        sum_probs += iterator * iterator

    gini = 1 - sum_probs
    return gini

gini_list = np.zeros(shape=(len(candidate_item_list), 2), dtype=object)
j = 0

# Compute Gini for each item
for i_x in candidate_item_list:
    item_i_df = data_pd[data_pd['item_id'] == i_x]
    gini_list[j] = [i_x,gini(item_i_df['raw_ratings'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId' : gini_list[:,0],'gini' : gini_list[:,1]}
gini_items_df = pd.DataFrame(to_df)
gini_items_df.sort_values(by='gini',inplace=True,ascending=False)

print('Computed Gini scoring')

# Prepare item Gini scores for merging
gini_items_df2 = gini_items_df.sort_values(by='itemId')
gini_items_df2.set_index(keys='itemId',inplace=True)
item_freq_df.set_index(keys='item_id',inplace=True)

# Merge frequencies and entropies
popgini_items_df = pd.concat([item_freq_df,gini_items_df2],axis=1,join='inner')

# Set weights for the popgini score
weight_popularity = 0.9
weight_gini = 1

# Compute popgini score
popgini_items_df['popgini'] = weight_popularity*np.log10(popgini_items_df['freq'])+weight_gini*popgini_items_df['gini']
popgini_items_df.sort_values(by='popgini',inplace=True,ascending=False)
item_freq_df.reset_index(inplace=True)
popgini_items_df.reset_index(inplace=True)
popgini_items_df.rename(columns={'index': 'item_id'}, inplace=True)

print('Computed PopGini scoring')

# In[13]:

### RESULTS

# Set number of items to show to the cold user
items_to_be_shown = [10, 25, 50, 100]    

# Create dataframe for results
ranking_strategies = ['Random strategy', 'Risky Y-change strategy', 'Moderate Y-change strategy', 'Conservative Y-change strategy', 'Risky CV-based strategy', 'Moderate CV-based strategy', 'Conservative CV-based strategy', 'PopGini strategy']
results_df = pd.DataFrame(columns=[ranking_strategies], index=[items_to_be_shown])
results_df_dich = pd.DataFrame(columns=[ranking_strategies], index=[items_to_be_shown])

# Compute results for each strategy and amount of items under consideration
for nr_of_shown_items in items_to_be_shown:
    print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))
    
    ### RANDOM STRATEGY ###
    # Select [nr_of_shown_items] items at random
    random.seed(1)
    random_items = random.sample((threshold_item_df.tolist()), nr_of_shown_items)
    random_items = np.array(random_items)
    print('Computed ranking using random strategy')
    
    ### Y-CHANGE STRATEGY ###
    # Select [nr_of_shown_items] items with largest Y-change
    y_change_items_risky = y_change_items_risky_df.head(nr_of_shown_items)['item_id']
    y_change_items_risky = np.array(y_change_items_risky)
    y_change_items_moderate = y_change_items_moderate_df.head(nr_of_shown_items)['item_id']
    y_change_items_moderate = np.array(y_change_items_moderate)
    y_change_items_conservative = y_change_items_conservative_df.head(nr_of_shown_items)['item_id']
    y_change_items_conservative = np.array(y_change_items_conservative)
    print('Computed ranking using Y-change strategy')
    
    ### CV-BASED STRATEGY ###
    # Select [nr_of_shown_items] items with smallest CV-based score 
    cv_based_items_risky = cv_based_items_risky_df.head(nr_of_shown_items)['item_id']
    cv_based_items_risky = np.array(cv_based_items_risky)
    cv_based_items_moderate = cv_based_items_moderate_df.head(nr_of_shown_items)['item_id']
    cv_based_items_moderate = np.array(cv_based_items_moderate)
    cv_based_items_conservative = cv_based_items_conservative_df.head(nr_of_shown_items)['item_id']
    cv_based_items_conservative = np.array(cv_based_items_conservative)
    print('Computed ranking using CV-based strategy')
    
    ### POPGINI STRATEGY ###
    # Select [nr_of_shown_items] items with largest PopGini score 
    popgini_items = popgini_items_df.head(nr_of_shown_items)['item_id']
    popgini_items = np.array(popgini_items)
    print('Computed ranking using PopGini strategy')
    
    # Set number of shown items
    number_of_shown_items = str(nr_of_shown_items)
    
    print('Computing results...')
    
    ### RANDOM STRATEGY ###
    
    ranking_strategy = 'Random strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(random_items))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(random_items))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(random_items))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    print('RMSE computed for random strategy')
    
    ### RISKY Y-CHANGE STRATEGY ###
    
    ranking_strategy = 'Risky Y-change strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(y_change_items_risky))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(y_change_items_risky))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(y_change_items_risky))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    ranking_strategy = 'Moderate Y-change strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(y_change_items_moderate))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(y_change_items_moderate))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(y_change_items_moderate))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    ranking_strategy = 'Conservative Y-change strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(y_change_items_conservative))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(y_change_items_conservative))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(y_change_items_conservative))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    print('RMSE computed for Y-change strategy')
    
    ### CV-BASED STRATEGY ###
    
    ranking_strategy = 'Risky CV-based strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(cv_based_items_risky))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(cv_based_items_risky))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(cv_based_items_risky))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    ranking_strategy = 'Moderate CV-based strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(cv_based_items_moderate))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(cv_based_items_moderate))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(cv_based_items_moderate))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    ranking_strategy = 'Conservative CV-based strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(cv_based_items_conservative))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(cv_based_items_conservative))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(cv_based_items_conservative))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    print('RMSE computed for CV-based strategy')
    
    ### POPGINI STRATEGY ###
    
    ranking_strategy = 'PopGini strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(popgini_items))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(popgini_items))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(popgini_items))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
        
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse.item()
    means = results_df.mean(axis=0)
    
    print('RMSE computed for PopGini strategy')
    print('Completed evaluation')
