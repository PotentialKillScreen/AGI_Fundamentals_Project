import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class LinearQNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearQNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save_model(self, filename='model.pth'):
        model_dir = './model'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        file_path = os.path.join(model_dir, filename)
        torch.save(self.state_dict(), file_path)


class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        pred_q_values = self.model(state)

        target_q_values = pred_q_values.clone()
        for idx in range(len(done)):
            q_update = reward[idx]
            if not done[idx]:
                q_update += self.gamma * torch.max(self.model(next_state[idx]))

            target_q_values[idx][torch.argmax(action[idx]).item()] = q_update
    
        self.optimizer.zero_grad()
        loss = self.loss_fn(target_q_values, pred_q_values)
        loss.backward()
        self.optimizer.step()