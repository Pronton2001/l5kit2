import torch
from torch.utils.data import Dataset, DataLoader

from pref_db import PrefDB

class PairwiseTrajDataset(Dataset):
    def __init__(self, pairwise_traj):
        self.pairwise_traj = pairwise_traj
        
    def __len__(self):
        return len(self.pairwise_traj)
    
    def __getitem__(self, idx):
        traj1, traj2, pref = self.pairwise_traj[idx]
        traj1 = torch.tensor(traj1, dtype=torch.float32)
        traj2 = torch.tensor(traj2, dtype=torch.float32)
        pref = torch.tensor(pref, dtype=torch.float32)
        return traj1, traj2, pref

pairwise_traj = [
    ([[1, 2], [2, 3], [3, 4]], [[2, 3], [3, 4], [4, 5]], (0.7, 0.3)),
    ([[2, 3], [3, 4], [4, 5]], [[1, 2], [2, 3], [3, 4]], (0.2, 0.8)),
    ([[3, 4], [4, 5], [5, 6]], [[4, 5], [5, 6], [6, 7]], (0.5, 0.5)),
]
pairwise_traj_dataset = PairwiseTrajDataset(pairwise_traj)
dataloader = DataLoader(pairwise_traj_dataset, batch_size=32, shuffle=True)
import torch
import torch.nn as nn
import torch.optim as optim
# class RewardModel(nn.Module):
#     def __init__(self, num_inputs, hidden_size):
#         super(RewardModel, self).__init__()
#         self.fc1 = nn.Linear(num_inputs, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 2)

#     def forward(self, input):
#         x = torch.relu(self.fc1(input))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
import torch.nn as nn

class BTMultiLabelModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BTMultiLabelModel, self).__init__()
        # (num_state + num_action) x 32, 32x16, 16x(2 probs)
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, traj1, traj2):
        # Concatenate all state-action pairs from both trajectories
        # print('traj1', traj1.shape) # 3,3,2
        x = torch.cat((traj1, traj2)) # 2* num_steps = 2* 3= 6, num_batch = 3, num state + action = 2
        # print('concate', x.shape) # 6,3,2

        # Pass through fully connected layers with ReLU activation
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)

        # Apply softmax to get probability distribution over labels
        # reshape to (2 * numsteps * num_batch)
        
        # print('before', x.shape)
        # x = x.view(-1,2)
        rs = x.sum(axis = 0)
        # print('sum', rs)
        # rs = rs.view(1,-1)
        # print('rehap', rs.shape)
        pred = self.softmax(rs) # softmax
        # print('softmax', pred)
        # print(pred)
        # Return predicted probability of trajectory 1 winning
        return pred


prefs_train= PrefDB(maxlen=5).load('/home/pronton/rl/l5kit/src/pref/preferences/5.pkl.gz')
pairwise_traj_dataset = PairwiseTrajDataset(pairwise_traj)
dataloader = DataLoader(pairwise_traj_dataset, batch_size=32, shuffle=True)
import matplotlib.pyplot as plt
import numpy as np
k1, _, _ = prefs_train.prefs[0]
assert np.array(prefs_train.segments[k1]).shape == (18,2), 'Error shape:' + str(np.array(prefs_train[k1]).shape) + 'vs (18,2)'
for i in range(len(prefs_train)):
    k1, k2, _ = prefs_train.prefs[i]
    for k in [k1, k2]:
        assert np.array(prefs_train.segments[k][0][0]['image']).shape == (7,112,112), 'error shape'
        for y in range(np.array(prefs_train.segments[k]).shape[0]):
            plt.imshow(prefs_train.segments[k][y][0]['image'][0]) # k, 0, 0
            plt.show()
        

# del self.prefs[n]
# print(len(prefs_train))
# print(prefs_train.prefs)
# s1s = [prefs_train.segments[k1] for k1, k2, pref, in batch]
# s2s = [prefs_train.segments[k2] for k1, k2, pref, in batch]
# prefs = [pref for k1, k2, pref, in batch]
# print(prefs_train[0])

# pairwise_traj = 
exit()
# Initialize reward model
reward_model = BTMultiLabelModel(state_dim=1, action_dim=1)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(reward_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10000):
    for traj1, traj2, pref in dataloader:
        # Zero out optimizer gradients
        optimizer.zero_grad()

        # Get predicted probability of trajectory 1 winning
        p1_pred = reward_model(traj1, traj2) # 2, num_step, state,action

        # Convert preference label to binary tensor
        # p1_pred = 6,3 (2 * num)
        # print(pref) # 3,
         
        # mu1 = pref # 3,1
        # mu2 = 1 - pref# 3,1

        pref_tensor = torch.tensor(pref)

        # Compute binary cross-entropy loss with preference label
        loss = criterion(p1_pred, pref_tensor)

        # Backpropagate and update parameters
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print('Iteration %d, loss = %.4f' % (epoch, loss.item()))