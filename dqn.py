import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import game2048_env
import csv
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter


# Qネットワーク
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # 隠れ層(y=Wx+bのやつ)
        self.fc1 = nn.Linear(state_dim, 128)    # 線形結合1
        self.fc2 = nn.Linear(128, 128)          # 線形結合2
        self.fc3 = nn.Linear(128, action_dim)   # 線形結合3

    # 順伝搬
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN_QNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)

        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# バッファー(この中からデータを学習)
class RepalyBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ダブルDQN(QネットワークとTargetネットワーク)
class DoubleDQN:
    def __init__(self, state_dim, action_dim):
        self.device = "cuda"

        self.q_net = CNN_QNetwork(action_dim).to(self.device)
        self.target_net = CNN_QNetwork(action_dim).to(self.device)

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=1e-4
        )

        self.buffer = RepalyBuffer()

        self.gamma = 0.99
        self.batch_size = 128

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(4)

        state = preprocess_state(state)
        state = state.unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.q_net(state)

        return q.argmax().item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 現在のQ値
        q_values = self.q_net(states)
        # print(q_values.shape)
        # print(actions.shape)
        # print(states.shape)
        q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 行動選択(main)
        next_actions = self.q_net(next_states).argmax(1)

        # 評価(target)
        next_q = self.target_net(next_states)\
            .gather(1, next_actions.unsqueeze(1))\
            .squeeze(1)

        target = rewards + self.gamma * next_q * (1 - dones)
        target = target.to(self.device)

        loss = F.smooth_l1_loss(q, target.detach())
        # print("loss: ", loss.item())
        # print("q_mean: ", q.mean().item())
        # print("target mean: ", target.mean().item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )


def preprocess_state(state):
    state = np.array(state, dtype=np.float32)

    state = torch.FloatTensor(state)

    state = state.view(4, 4)

    state = state.unsqueeze(0)

    return state


env = game2048_env.Env2048()

agent = DoubleDQN(16, 4)

epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.9995

total_steps = 0

results = []

Writer = SummaryWriter('logs/2048_experiment_6')

for episode in range(15000):
    state = env.reset()

    done = False

    total_reward = 0.0

    steps = 0

    while not done:
        state_t = preprocess_state(state)
        action = agent.select_action(state_t, epsilon)

        next_state, reward, done, score, max = env.step(action)

        agent.buffer.push(
            state_t,
            torch.tensor(action),
            torch.tensor(reward, dtype=torch.float),
            torch.FloatTensor(preprocess_state(next_state)),
            torch.tensor(done, dtype=torch.float)
        )

        total_reward += reward
        agent.train()
        state = next_state
        steps += 1
        total_steps += 1

    if total_steps % 1000 == 0:
        agent.update_target()

    results.append([episode, total_reward, score])
    Writer.add_scalar('Reward/Total', total_reward, episode)
    Writer.add_scalar('Score/Train', score, episode)

    print(f"Episode: {episode}, total_reward {total_reward:.2f}, score {score}, epsilon {epsilon:.4f}, steps {steps}, max {max}")

    epsilon *= epsilon_decay
    if epsilon < epsilon_min:
        epsilon = 0.5

with open('data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'total_reward', 'score'])
    writer.writerows(results)

print("done")
