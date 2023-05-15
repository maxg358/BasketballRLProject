import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import os
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from nba_simulation import *

class AdvantageActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4, continuous=False):
        super(AdvantageActorCritic, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size).to(self.device)
        self.critic_linear2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.critic_linear3 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.critic_linear4 = nn.Linear(hidden_size, 1).to(self.device)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size).to(self.device)
        self.actor_linear2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.actor_linear3 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.actor_linear4 = nn.Linear(hidden_size, num_actions).to(self.device)
        logstds_param = nn.Parameter(torch.full((num_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
        self.cont = continuous
    
    def forward(self, state):
        state = Variable(state.float().unsqueeze(0))
        value = F.tanh(self.critic_linear1(state))
        value = F.tanh(self.critic_linear2(value))
        value = F.tanh(self.critic_linear3(value))
        value = self.critic_linear4(value)
        
        policy_dist = F.tanh(self.actor_linear1(state))
        policy_dist = F.tanh(self.actor_linear2(policy_dist))
        policy_dist = F.tanh(self.actor_linear3(policy_dist))
        if self.cont:
            policy_dist = self.actor_linear4(policy_dist)
            stds = torch.clamp(self.logstds.exp(), 1e-3, 50).to(self.device)
            policy_dist = torch.distributions.Normal(policy_dist, stds)
        else:
            policy_dist = F.softmax(self.actor_linear4(policy_dist), dim=1)

        return value, policy_dist

def training_loop(episodes):
    
    # Constants
    GAMMA = 0.99
    num_inputs = 55
    hidden_size = 64
    non_ball_lr = 5e-4
    ball_lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_lengths = []
    average_lengths = []
    all_rewards = []
    Dentropy_term = 0
    Oentropy_term = 0
    Bentropy_term = 0
    movement_space = 2
    ball_handler_space = 6
    # model and environment inits
    game = NBAGymEnv(random=False)
    offense_model = AdvantageActorCritic(num_inputs, movement_space, hidden_size, continuous=True)
    defense_model = AdvantageActorCritic(num_inputs, movement_space, hidden_size, continuous=True)
    ball_model = AdvantageActorCritic(num_inputs, ball_handler_space, hidden_size)
    offense_optimizer = optim.Adam(offense_model.parameters(), lr=non_ball_lr)
    defense_optimizer = optim.Adam(defense_model.parameters(), lr=non_ball_lr)
    ball_optimizer = optim.Adam(ball_model.parameters(), lr=ball_lr)
    
    # number of games loop
    all_rewards = []
    
    for episode in tqdm(range(episodes)):
        actions = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
        Dlog_probs = []
        Dvalues = []
        Olog_probs = []
        Ovalues = []
        Blog_probs = []
        Bvalues = []
        Orewards = []
        Brewards = []
        Drewards = []
        done = False
        # within game loop
        while not done:
            collect_breward = False
            for i, agent in enumerate(game.players):
                player_state = game.shift_perspective(agent.player) # shift perspective of player
                player_state = torch.from_numpy(player_state).float().to(device) # convert to tensor
                if agent.team == 0:
                    # if agent is in posession of ball
                    if agent.posession:
                        # forward pass
                        value, policy_dist = ball_model.forward(player_state)
                        value = value.detach().cpu().numpy()[0,0]
                        dist = policy_dist.detach().cpu().numpy() 
                        action = np.random.choice(ball_handler_space, p=np.squeeze(dist))
                        log_prob = torch.log(policy_dist.squeeze(0)[action])
                        entropy = -np.sum(np.mean(dist) * np.log(dist))
                        # if dribbling using the offense model to dictate movement
                        if action == agent.player:
                            value, policy_dist = offense_model.forward(player_state)
                            value = value.detach().cpu().numpy()[0,0]
                            action = policy_dist.sample().detach().cpu().data.numpy()[0]
                            log_prob = policy_dist.log_prob(torch.tensor(action).to(device))
                            entropy = policy_dist.entropy().mean()
                            actions[i] = agent.act(game.state, curr_vector=action, prev_vector=actions[i])
                            Ovalues.append(value)
                            Olog_probs.append(torch.squeeze(log_prob))
                            Oentropy_term += entropy
                        # otherwise perform action in background
                        else:
                            actions[i] = agent.act(game.state, curr_vector=action, prev_vector=actions[i])
                            collect_breward = True
                            Bvalues.append(value)
                            Blog_probs.append(log_prob)
                            Bentropy_term += entropy
                    # if not in posession use normal offense model to dictate movement
                    else:
                        value, policy_dist = offense_model.forward(player_state)
                        value = value.detach().cpu().numpy()[0,0]
                        action = policy_dist.sample().detach().cpu().data.numpy()[0]
                        log_prob = policy_dist.log_prob(torch.tensor(action).to(device))
                        entropy = policy_dist.entropy().mean()
                        actions[i] = agent.act(game.state, curr_vector=action, prev_vector=actions[i])

                        Ovalues.append(value)
                        Olog_probs.append(torch.squeeze(log_prob))
                        Oentropy_term += entropy
                # if not on team 0, use defense model to dictate movement
                else:
                    value, policy_dist = defense_model.forward(player_state)
                    value = value.detach().cpu().numpy()[0,0]
                    action = policy_dist.sample().detach().cpu().data.numpy()[0]
                    log_prob = policy_dist.log_prob(torch.tensor(action).to(device))
                    entropy = policy_dist.entropy().mean()
                    actions[i] = agent.act(game.state, curr_vector=action, prev_vector=actions[i])
                    
                    Dvalues.append(value)
                    Dlog_probs.append(torch.squeeze(log_prob))
                    Dentropy_term += entropy

            Breward, Oreward, Dreward, done = game.step(actions)
            Orewards += Oreward
            if collect_breward:
                Brewards.append(Breward)
            Drewards += Dreward

        all_rewards.append(np.sum(Orewards) + np.sum(Brewards) + np.sum(Drewards))

        state = torch.from_numpy(game.state).float().to(device)
        DQval, _ = defense_model.forward(state)
        DQval = DQval.detach().cpu().numpy()[0,0]
        OQval, _ = offense_model.forward(state)
        OQval = OQval.detach().cpu().numpy()[0,0]
        BQval, _ = ball_model.forward(state)
        BQval = BQval.detach().cpu().numpy()[0,0]
        DQvals = np.zeros_like(Dvalues)
        OQvals = np.zeros_like(Ovalues)
        BQvals = np.zeros_like(Bvalues)

        for t in reversed(range(len(Drewards))):
            DQval = Drewards[t] + GAMMA * DQval
            DQvals[t] = DQval
        for t in reversed(range(len(Orewards))):
            OQval = Orewards[t] + GAMMA * OQval
            OQvals[t] = OQval
        for t in reversed(range(len(Brewards))):
            BQval = Brewards[t] + GAMMA * BQval
            BQvals[t] = BQval

        #update actor critic
        Dvalues = torch.FloatTensor(Dvalues)
        DQvals = torch.FloatTensor(DQvals)
        Dlog_probs = torch.stack(Dlog_probs).cpu()
        Dadvantage = DQvals - Dvalues
        Dactor_loss = (torch.mul(-Dlog_probs.T, Dadvantage).T).mean(dtype=torch.float64)
        Dcritic_loss = 0.5 * Dadvantage.pow(2).mean()
        Dac_loss = Dactor_loss + Dcritic_loss + 0.001 * Dentropy_term.cpu()
        defense_optimizer.zero_grad()
        Dac_loss.backward(retain_graph=True)
        defense_optimizer.step()

        Ovalues = torch.FloatTensor(Ovalues)
        OQvals = torch.FloatTensor(OQvals)
        Olog_probs = torch.stack(Olog_probs).cpu()
        Oadvantage = OQvals - Ovalues
        Oactor_loss = (torch.mul(-Olog_probs.T, Oadvantage).T).mean(dtype=torch.float64)
        Ocritic_loss = 0.5 * Oadvantage.pow(2).mean()
        Oac_loss = Oactor_loss + Ocritic_loss + 0.001 * Oentropy_term.cpu()
        offense_optimizer.zero_grad()
        Oac_loss.backward(retain_graph=True)
        offense_optimizer.step()
        
        if len(Brewards) != 0:
            Bvalues = torch.FloatTensor(Bvalues)
            BQvals = torch.FloatTensor(BQvals)
            Blog_probs = torch.stack(Blog_probs).cpu()
            Badvantage = BQvals - Bvalues
            Bactor_loss = (-Blog_probs * Badvantage).mean()
            Bcritic_loss = 0.5 * Badvantage.pow(2).mean()
            Bac_loss = Bactor_loss + Bcritic_loss + 0.001 * Bentropy_term
            ball_optimizer.zero_grad()
            Bac_loss.backward(retain_graph=True)
            ball_optimizer.step()

        game.reset()
    game.render()
    return all_rewards

all_rewards = training_loop(2)
smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
smoothed_rewards = [elem for elem in smoothed_rewards]
plt.plot(all_rewards)
plt.plot(smoothed_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward vs Episode')
plt.savefig('reward.png')

