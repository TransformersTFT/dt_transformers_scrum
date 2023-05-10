import numpy as np
import pickle

# Define dataset parameters
n_episodes = 100
ep_len = 200
state_dim = 11
action_dim = 4

# Initialize dataset
dataset = {'observations': np.zeros((n_episodes, ep_len, state_dim)),
           'actions': np.zeros((n_episodes, ep_len, action_dim)),
           'rewards': np.zeros((n_episodes, ep_len)),
           'terminals': np.zeros((n_episodes, ep_len)),
           'next_observations': np.zeros((n_episodes, ep_len, state_dim))}

# Fill dataset with random data
for ep in range(n_episodes):
    for t in range(ep_len):
        dataset['observations'][ep, t, :] = np.random.rand(state_dim)
        dataset['actions'][ep, t, :] = np.random.rand(action_dim)
        dataset['rewards'][ep, t] = np.random.rand()
        dataset['terminals'][ep, t] = np.random.choice([0, 1])
        dataset['next_observations'][ep, t, :] = np.random.rand(state_dim)

# Save dataset as pickle file
with open('env_va-dataset-v3.pkl', 'wb') as f:
    pickle.dump(dataset, f)