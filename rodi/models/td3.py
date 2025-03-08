import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from rodi.models.base import BaseModel


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """
        Initialize the actor network

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            max_action: Maximum action value
        """
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layer1(state))
        a = F.relu(self.layer2(a))
        a = (
            torch.sigmoid(self.layer3(a)) * self.max_action
        )  # Ensure actions are between 0 and max_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic network

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        super(Critic, self).__init__()

        # Q1 architecture
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

        # Q2 architecture
        self.layer4 = nn.Linear(state_dim + action_dim, 400)
        self.layer5 = nn.Linear(400, 300)
        self.layer6 = nn.Linear(300, 1)

    def forward(self, state, action):
        """
        Forward pass through both critic networks
        """
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer1(sa))
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        q2 = F.relu(self.layer4(sa))
        q2 = F.relu(self.layer5(q2))
        q2 = self.layer6(q2)

        return q1, q2

    def Q1(self, state, action):
        """
        Forward pass through just the first critic network
        """
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer1(sa))
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        return q1


class TD3(BaseModel):
    """
    Twin Delayed Deep Deterministic Policy Gradients (TD3) agent
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        """
        Initialize the TD3 agent

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            max_action: Maximum action value
            device: Device to run the model on (cpu or cuda)
            discount: Discount factor
            tau: Target network update rate
            policy_noise: Noise added to target policy during critic update
            noise_clip: Range to clip target policy noise
            policy_freq: Frequency of delayed policy updates
        """
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        noise = (torch.randn_like(action) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )

        next_action = (self.actor_target(next_state) + noise).clamp(0, self.max_action)

        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=self.device)
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer", map_location=self.device)
        )
        self.actor_target = copy.deepcopy(self.actor)
