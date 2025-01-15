from model_critic import Critic
from model_actor import Actor
from Buffer import Buffer
import torch
import gym
import torch.nn.functional as F
import numpy as np
import time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf


""" 
To clean the logs ////! don't change the path in rmtree
import shutil
shutil.rmtree('logs/', ignore_errors=True)
"""
env_name = 'Pendulum-v1'  # 'Pendulum-v1' #LunarLanderContinuous-v2

# Register the environment
"""
gym.envs.register(
    id=env_name+'v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=250,      # MountainCar-v0 uses 200
    reward_threshold=-110.0,
)
"""

env = gym.make(env_name)  # other_games to benchmarck our code

log_dir = 'logs/' + env_name + str(time.time())
summary_writer = SummaryWriter(log_dir=log_dir)


class Agent:
    def __init__(self, env=env, total_eps=1e2, batch_size=256, alpha=0.2, max_mem_length=1e6, tau=0.1,
                 Q_lr=3e-4, policy_lr=3e-4, alpha_lr=3e-4, gamma=0.9, summary_writer=summary_writer):
        self.env = env
        self.total_eps = int(total_eps)
        self.steps = 0
        self.total_scores = []
        self.batch_size = batch_size
        self.tau = tau
        self.buffer = Buffer(int(max_mem_length))
        self.gamma = gamma

        self.Actor = Actor(
            env.observation_space.shape[0], env.action_space.shape[0])
        self.Q1 = Critic(
            [env.observation_space.shape[0], env.action_space.shape[0]])
        self.Q2 = Critic(
            [env.observation_space.shape[0], env.action_space.shape[0]])
        self.target_Q1 = Critic(
            [env.observation_space.shape[0], env.action_space.shape[0]])
        self.target_Q1.load_state_dict(self.Q1.state_dict())
        self.target_Q2 = Critic(
            [env.observation_space.shape[0], env.action_space.shape[0]])
        self.target_Q2.load_state_dict(self.Q2.state_dict())
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.entropy_target = torch.scalar_tensor(
            -env.action_space.shape[0], dtype=torch.float64)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=Q_lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=Q_lr)
        self.Actor_optimizer = optim.Adam(
            self.Actor.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=alpha_lr)

        self.summary_writer = summary_writer

    def run(self):
        for ep in range(self.total_eps):
            print(f'=========== Epoch n°{ep}==============\n')
            total_score = 0
            done = False
            state = self.env.reset()[0]
            # state = torch.tensor(state, dtype=torch.float32)
            current_step = 0
            while (done is False) and (current_step < 300):
                # print(f"===StepN°{current_step}===")
                current_step += 1
                self.steps += 1
                state = torch.tensor(state, dtype=torch.float32)
                action, log_pi = self.Actor.sample_action(state)
                # interpolate to get action is form for env's space
                action_flatten = np.squeeze(action, axis=0)
                env_action = torch.tensor(self.env.action_space.low, dtype=torch.float32) + \
                    (action_flatten + 1) / 2 * \
                    (torch.tensor(self.env.action_space.high, dtype=torch.float32) -
                     torch.tensor(self.env.action_space.low, dtype=torch.float32))
                next_state, reward, done, info, _ = self.env.step(
                    env_action.numpy())
                total_score += reward
                # Pass only array into the buffer !!
                self.buffer.add((state.numpy(), action.numpy(),
                                reward, next_state, 1 - done))
                state = next_state
                if len(self.buffer.buffer) > self.batch_size:
                    self.learn()
            with self.summary_writer:
                self.summary_writer.add_scalar(
                    "total score", total_score, self.steps)
            self.total_scores.append(total_score)

    def learn(self):
        batch = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, not_done = zip(*batch)
        state = torch.tensor(state, dtype=torch.float32, requires_grad=True)
        action = torch.tensor(action, dtype=torch.float32, requires_grad=True)
        action = torch.reshape(
            action, (action.size()[0], env.action_space.shape[0]))
        reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)
        reward = torch.reshape(reward, (reward.size()[0], 1))
        next_state = torch.tensor(next_state, dtype=torch.float32)
        not_done = torch.tensor(not_done, dtype=torch.float32)
        not_done = torch.reshape(not_done, (not_done.size()[0], 1))

        # Learn Q1 Q2
        with torch.no_grad():
            next_action, next_log_pi = self.Actor.sample_action(next_state)
            next_Q1 = self.target_Q1(next_state, next_action)
            next_Q2 = self.target_Q2(next_state, next_action)
            next_Q = torch.min(next_Q1, next_Q2) - self.alpha * next_log_pi
            target_Q = reward + self.gamma * not_done * next_Q

        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        Q1 = self.Q1(state, action)
        Q2 = self.Q2(state, action)
        Q1_loss = F.mse_loss(Q1, target_Q)
        Q2_loss = F.mse_loss(Q2, target_Q)
        Q1_loss.backward()
        Q2_loss.backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()

        with self.summary_writer:
            self.summary_writer.add_scalar(
                "Q1 loss", Q1_loss.detach().numpy(), self.steps)
            self.summary_writer.add_scalar(
                "Q2 loss", Q2_loss.detach().numpy(), self.steps)

        # Learn Policy
        self.Actor_optimizer.zero_grad()
        action, log_pi = self.Actor.sample_action(state)
        Q1 = self.Q1(state, action)
        Q2 = self.Q2(state, action)
        Q = torch.min(Q1, Q2)
        policy_loss = (-self.alpha.detach() * log_pi - Q).mean()
        print(f"Policy loss = {policy_loss}\n")
        with self.summary_writer:
            self.summary_writer.add_scalar(
                "policy loss", policy_loss, self.steps)
        policy_loss.backward()
        self.Actor_optimizer.step()

        # Learn alpha (temperature)
        self.alpha_optimizer.zero_grad()
        alpha_loss = -(self.alpha * (- log_pi + self.entropy_target)).mean()
        with self.summary_writer:
            self.summary_writer.add_scalar(
                "alpha loss", alpha_loss, self.steps)
        alpha_loss.backward(inputs=[self.alpha])
        self.alpha_optimizer.step()
        # print(f"alpha = {self.alpha}\n")

        self.update_targets()

    def update_targets(self):
        # Update weights of soft-value function (V_psi in the paper) by moving average
        for target_param, param in zip(self.target_Q1.parameters(), self.Q1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
