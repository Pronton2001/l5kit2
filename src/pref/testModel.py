import random
import torch
import torch.nn as nn
import torch.optim as optim

class RewardModel(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, input):
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate example dataset
dataset = [
    (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]), (0.7, 0.3)),
    (torch.tensor([2, 3, 4]), torch.tensor([1, 2, 3]), (0.2, 0.8)),
    (torch.tensor([3, 4, 5]), torch.tensor([4, 5, 6]), (0.5, 0.5)),
    (torch.tensor([4, 5, 6]), torch.tensor([3, 4, 5]), (0.9, 0.1)),
    (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), (0.3, 0.7)),
    (torch.tensor([4, 5, 6]), torch.tensor([1, 2, 3]), (0.8, 0.2)),
]

# Initialize reward model
reward_model = RewardModel(num_inputs=3, hidden_size=64)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

# Train reward model using pairwise comparisons
for i in range(10000):
    # Randomly select a pair of trajectories from the dataset
    traj1, traj2, pref = dataset[random.randint(0, len(dataset) - 1)]

    # Compute reward prediction for each trajectory
    pred = reward_model(torch.stack([traj1.float(), traj2.float()]))
    # print(pred, pred.shape)


    # Compute loss and update model parameters
    target = torch.tensor([0 if pref[0] > pref[1] else 1])
    # print(target)
    loss = criterion(pred.view(1, 2), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 iterations
    if i % 100 == 0:
        print('Iteration %d, loss = %.4f' % (i, loss.item()))

