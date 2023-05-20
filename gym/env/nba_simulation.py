import math
import warnings
import numpy as np
import gym
import random
from math import sqrt
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
import os
import time
sys.path.append('..')
from copy import deepcopy
# from utils.data import *
# from ..utils import *
# from ..vis import *

class Agent():
    def __init__(self, id, posession=False, team=0):
        super(Agent, self).__init__()
        self.player = id
        self.movement_space = np.array([0.0, 0.0])
        self.posession = posession
        #Dribble (pass to yourself), Pass to 4 others, Shoot (a pass to the rim)
        self.ball_handler_space = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        self.team = team

class ModelAgent(Agent):
    def __init__(self, id, posession=False, team=0):
        super(ModelAgent, self).__init__(id, posession, team)
        self.movement_space = np.array([0.0, 0.0])
        self.max_player_speed = 2

    def ball_action(self, state, choice):
        # if pass to yourself then dribble
        self.ball_handler_space = np.zeros(6)
        self.ball_handler_space[choice] = 1.0
        return deepcopy(self.ball_handler_space)
        
    def act(self, state, curr_vector=[0., 0.], prev_vector=[0., 0.], min_val=-2, max_val=2, decimal_places=1):
        # if they have the ball perform ball action
        if self.posession == True and type(curr_vector) == int:
            a = self.ball_action(state, curr_vector)
            if np.argmax(a) != self.player:
                # no movement cause ball movement
                return np.array([0.0, 0.0])
        # get velocity in x and y direction
        action = curr_vector
        # add to previous vector
        action += prev_vector
        # if larger than max speed then cap it at that
        if np.linalg.norm(action) > self.max_player_speed:
            action /= np.linalg.norm(action)
            action *= self.max_player_speed
        return action


class RandomAgent(Agent):
    def __init__(self, id, posession=False, team=0):
        super(RandomAgent, self).__init__(id, posession, team)
        self.movement_space = np.array([0.0, 0.0])
        self.max_player_speed = 2

    def ball_action(self):
        
        # weighted probability of passing/shooting vs dribbling
        
        weights = [.08, .08, .08, .08, .08, .08]
        weights[self.player] = .6
        choice = random.choices(range(6), weights=weights)[0]
        # if pass to yourself then dribble
        if choice == self.player:
            self.ball_handler_space = np.zeros(6)
            self.ball_handler_space[choice] = 1.0
            print(self.player, 'dribble choice')
            return [None]
        else:
            print(self.player,'other choice', choice)
            self.ball_handler_space = np.zeros(6)
            self.ball_handler_space[choice] = 1.0
            return deepcopy(self.ball_handler_space)
        
    def act(self, prev_vector=[0., 0.], min_val=-2, max_val=2, decimal_places=1):
        # if they have the ball perform ball action
        if self.posession == True:
            a = self.ball_action()
            if a[0] != None:
                # no movement cause ball movement
                return np.array([0.0, 0.0])
        # get velocity in x and y direction
        float1 = round(random.uniform(min_val, max_val), decimal_places)
        float2 = round(random.uniform(min_val, max_val), decimal_places)
        action = np.array([float1, float2])
        # add to previous vector
        action += prev_vector
        # if larger than max speed then cap it at that
        if np.linalg.norm(action) > self.max_player_speed:
            action /= np.linalg.norm(action)
            action *= self.max_player_speed
        return action

class NBAGymEnv(gym.Env):
    def __init__(self, random=True):
        super(NBAGymEnv, self).__init__()
        '''
            State: dimensions 0-39 are [2D position, 2D direction] + 10D posession
            + 2D ball position + 2D ball vector + 1D shot clock - 55 features
            Action: 2D movement for all players except ball handler
            Rewards: Rules for as follows
                Defense:
                    - +7 if in position to block
                    - +7 if in position to steal
                    - +15/distance to closest offensive player
                OnBall:
                    - -10 for turnover (stolen, blocked, or out of bounds)
                    - -2 to -40 for not taking a shot within 24 seconds
                    - +0 for every second the ball is in the air during shot
                    - +20 for made shot
                    - +0.5 for every completed pass
                    - +dist/3 to nearest defender
                Offense:
                    - +dist/3 to nearest defender
                    - +1 for being in position to catch pass
                    - +distance to nearest offensive player/10 for offensive spacing
                    - +distance to nearest defensive player/10 for being open

                
        '''
        # reset fields
        # Define action and observation spaces
        r_p = self.generate_points(hard_coded=not random)
        # 10 players’ info ( 2D position vectors + 2D direction vector) + 10D who has the ball + 2D ball position + 2D ball vector + 1D shot clock
        self.state = np.array([*r_p[0], 0.0, 0.0, *r_p[1], 0.0, 0.0, *r_p[2], 0.0, 0.0,
                                *r_p[3], 0.0, 0.0, *r_p[4], 0.0, 0.0, *r_p[5], 0.0, 0.0,
                                *r_p[6], 0.0, 0.0, *r_p[7], 0.0, 0.0, *r_p[8], 0.0, 0.0,
                                *r_p[9], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, *r_p[0], 0.0, 0.0, 24.0])
        self.player_posession = 0
        self.shot_dist = 0
        self.ball_state = 'DRIBBLING' # or 'PASSING' or 'SHOOTING' or 'MIDAIR PASS' or 'MIDAIR SHOT' or 'MADE SHOT' or 'MISSED SHOT' or 'STOLEN' or 'BLOCKED'
        if random:
            self.players = [RandomAgent(0, True), RandomAgent(1), RandomAgent(2), RandomAgent(3), RandomAgent(4), RandomAgent(5, team=1), RandomAgent(6, team=1), RandomAgent(7, team=1), RandomAgent(8, team=1), RandomAgent(9, team=1)]
        else:
            self.players = [ModelAgent(0, True), ModelAgent(1), ModelAgent(2), ModelAgent(3), ModelAgent(4), ModelAgent(5, team=1), ModelAgent(6, team=1), ModelAgent(7, team=1), ModelAgent(8, team=1), ModelAgent(9, team=1)]
        # not reset fields
        self.force_field = 3.25
        self.player_radius = 3.25
        self.ball_radius = 1.5
        self.time_increment = 0.25 #seconds per unit
        self.max_player_speed = .5
        self.ball_speed = 2
        self.basket_location = (6, 25)
        self.court_dims = (47, 50)
        self.all_states = [deepcopy(self.state)]
        self.all_actions = {'stolen': 0, 'blocked': 0, 'outofbounds': 0, 'no_shot': 0, 'made_shot': 0, 'missed_shot': 0}
        self.random = random
        # Initialize the play

    def generate_points(self, min_distance=2.5, num_points=10, box_width=47, box_height=50, hard_coded=False):
        
        def distance(p1, p2):
            return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def is_valid_point(new_point, points, min_distance):
            for point in points:
                if distance(new_point, point) < min_distance:
                    return False
            return True

        points = []
        if hard_coded:
            # offense
            points.append((6, 44))
            points.append((18, 40))
            points.append((30, 25))
            points.append((18, 10))
            points.append((6, 6))
            # defense
            points.append((6, 42))
            points.append((16, 38))
            points.append((28, 25))
            points.append((16, 12))
            points.append((6, 8))
            return points

        attempts = 0

        while len(points) < num_points and attempts < 1000:
            new_point = (random.uniform(2, box_width-2), random.uniform(2, box_height-2))

            if is_valid_point(new_point, points, min_distance):
                points.append(new_point)
            else:
                attempts += 1

        if attempts == 1000:
            print("Failed to generate the required number of points with the specified minimum distance.")
            return []
        #self.state[-5:-3] = self.state[0:2]
        
        return points

    def reset(self):
        r_p = self.generate_points(hard_coded=not self.random)
        self.state = np.array([*r_p[0], 0.0, 0.0, *r_p[1], 0.0, 0.0, *r_p[2], 0.0, 0.0,
                                *r_p[3], 0.0, 0.0, *r_p[4], 0.0, 0.0, *r_p[5], 0.0, 0.0,
                                *r_p[6], 0.0, 0.0, *r_p[7], 0.0, 0.0, *r_p[8], 0.0, 0.0,
                                *r_p[9], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, *r_p[0], 0.0, 0.0, 24.0])
        self.ball_state = 'DRIBBLING'
        self.player_posession = 0
        self.shot_dist = 0
        if self.random:
            self.players = [RandomAgent(0, True), RandomAgent(1), RandomAgent(2), RandomAgent(3), RandomAgent(4), RandomAgent(5, team=1), RandomAgent(6, team=1), RandomAgent(7, team=1), RandomAgent(8, team=1), RandomAgent(9, team=1)]
        else:
            self.players = [ModelAgent(0, True), ModelAgent(1), ModelAgent(2), ModelAgent(3), ModelAgent(4), ModelAgent(5, team=1), ModelAgent(6, team=1), ModelAgent(7, team=1), ModelAgent(8, team=1), ModelAgent(9, team=1)]
        

    def reward_defense(self, player_id):
        player_location = [self.state[4*player_id],self.state[4*player_id+1]]
        ball_location = [self.state[-5], self.state[-4]]
        dist_list = []
        reward = 0
        for i in range(5):
            dist_list.append(self.spacing_helper(player_id,i))
        min_dist = np.min(dist_list)
        min_dist_inverse = min_dist # The closer they area to offensive players, the higher the reward
        reward+=10/min_dist_inverse #placeholder
        if math.dist(player_location, ball_location) < 5 and self.ball_state != 'MIDAIR SHOT': #roughly in position to steal
            reward+=2 #placeholder
            if(self.ball_state == 'MIDAIR PASS' or self.ball_state == 'SHOOTING' or self.ball_state == 'PASSING'):
                reward+=5
        if self.ball_state == 'MADE SHOT':
            reward -= 5
        return reward
        
    def reward_offense_onball(self, player_id):
        '''
            This gets called when player previously had posession or if player 
            currently has posession
        '''
        if self.ball_state == 'MIDAIR SHOT':
            return 0
        if self.ball_state == 'MIDAIR PASS':
            return 0
        if(self.ball_state == 'MADE SHOT'):
            return 10 #made shot
        reward = 0
        if(self.ball_state =='STOLEN' or self.ball_state =='BLOCKED' or self.ball_state=='OB'):
            reward -= 5 #turnover
        if(self.state[-1]<5.0 and self.ball_state != 'MIDAIR SHOT' and self.ball_state != 'MADE SHOT' and self.ball_state != 'MISSED SHOT'):
            if self.state[-1] != 0:
               reward -= 1/self.state[-1]*10  #shooting with low shot clock
            else:
                reward -= 40
        if self.ball_state == 'PASSING':
            reward += 2
        if(self.player_posession != player_id and self.player_posession<5):
            reward+=3 # completed pass
        dist_list = []
        for i in range(5,10):
            dist_list.append(self.spacing_helper(player_id,i))
        min_dist = np.min(dist_list)
        #min_dist_inverse = 1/min_dist # The closer they area to defensive players, the lower the reward
        reward-=10/min_dist #placeholder
        basket_dist = math.dist(self.state[player_id*4: player_id*4+2], self.basket_location)
        # if near basket and your action isn't shooting get penalized
        if basket_dist < 6.5 and self.ball_state != 'MIDAIR SHOT' and self.ball_state != 'SHOOTING':
            reward -= 10
        reward += 15/self.state[-1] if self.state[-1] > 10  else 0
        return reward
    
    def reward_offense_offball(self, player_id):
        player_location = [self.state[4*player_id],self.state[4*player_id+1]]
        ball_location = [self.state[-5], self.state[-4]]
        dist_list = []
        reward = 0
        for i in range(5,10):
            dist_list.append(self.spacing_helper(player_id,i))
        min_dist = np.min(dist_list)
        #min_dist_inverse = 1/min_dist # The closer they area to defensive players, the lower the reward
        reward-=10/min_dist #placeholder
        if math.dist(player_location, ball_location)<2 and self.ball_state == 'MIDAIR PASS' and self.player_posession != player_id: #roughly in position to catch ball
            reward+=3 #placeholder
        offensive_dist_list = []
        defensive_dist_list = []
        for i in range(5):
            if(i == player_id):
                continue
            offensive_dist_list.append(self.spacing_helper(player_id,i))
        min_offensive_dist = np.min(offensive_dist_list)
        reward+=min_offensive_dist/10 #the larger the space between offensive players, the larger the reward
        return reward
    
    def spacing_helper(self, player_id_1, player_id_2):

        player_1_location = [self.state[4*player_id_1],self.state[4*player_id_1+1]]
        player_2_location = [self.state[4*player_id_2],self.state[4*player_id_2+1]]
        return math.dist(player_1_location, player_2_location)
        
    def movement(self, actions):
        # update player positions - get proposed positions for all and do out of bounds check,
        proposed_pos = np.zeros((10, 2))
        for i, act in enumerate(actions):
            # cap on movement speed
            self.state[i*4+2:i*4+4] += act
            if np.linalg.norm(self.state[i*4+2:i*4+4]) > self.max_player_speed:
                self.state[i*4+2:i*4+4] /= np.linalg.norm(self.state[i*4+2:i*4+4])
                self.state[i*4+2:i*4+4] *= self.max_player_speed
            proposed_pos[i] = self.state[i*4:i*4+2] + self.state[i*4+2:i*4+4]
            
        # collision check
        for i in range(10):
            player1_new_position = proposed_pos[i]

            for j in range(10):
                if i == j:
                    continue
                player2_new_position = proposed_pos[j]
                distance = np.linalg.norm(player1_new_position - player2_new_position)
                
                # If the players are closer than the minimum allowed distance, adjust their positions
                if distance < self.force_field:
                    if distance > 0:
                        direction = (player1_new_position - player2_new_position) / distance
                        overlap = self.force_field - distance

                        # Move both players away from each other to maintain the minimum distance
                        player1_new_position += direction * (overlap / 2)
                        player2_new_position -= direction * (overlap / 2)

                        # Update the proposed positions
                        proposed_pos[i] = player1_new_position
                        proposed_pos[j] = player2_new_position
                    else:
                        # If the players are on top of each other, move them back
                        # This is not ideal, but it's a rare case
                        proposed_pos[i] = self.state[i*4:i*4+2]
                        proposed_pos[j] = self.state[j*4:j*4+2]

        # Update the actual player positions after resolving collisions
        for i in range(10):
            # out of bounds check
            if proposed_pos[i][0] > self.court_dims[0] - self.force_field:
                proposed_pos[i][0] = self.court_dims[0] - self.force_field
            elif proposed_pos[i][0] < self.force_field:
                proposed_pos[i][0] = self.force_field
            if proposed_pos[i][1] > self.court_dims[1] - self.force_field:
                proposed_pos[i][1] = self.court_dims[1] - self.force_field
            elif proposed_pos[i][1] < self.force_field:
                proposed_pos[i][1] = self.force_field
            
            self.state[i*4:i*4+2] = proposed_pos[i]

    def ball_movement(self):
        # if someone is dribbling, set ball pos to that of player with posession, ball direction to player direction
        if self.ball_state == 'DRIBBLING':
            self.state[-5:-3] = self.state[self.player_posession*4:self.player_posession*4+2]
            self.state[-3:-1] = self.state[self.player_posession*4+2:self.player_posession*4+4]
        else:
            # if ball being passed to someone/thing
            self.state[-5:-3] += self.state[-3:-1]
            prev_player = self.player_posession
            if self.ball_state == 'PASSING' or self.ball_state  == 'MIDAIR PASS':
                self.ball_state = 'MIDAIR PASS'
                #self.player_posession= None
                #print(self.ball_state)
                for i in range(10):
                    
                    distance = np.linalg.norm(self.state[i*4:i*4+2] - self.state[-5:-3])
                    if distance < self.player_radius + self.ball_radius and i!= self.player_posession:
                        # if near defense, give 15% chance of stealing, otherwise let it go through
                        if i >= 5 and random.choices([0, 1], weights=(0.15, 0.85))[0] == 1:
                            print('not stolen')
                            continue
                        print('caught by', i, prev_player)
                        
                        self.player_posession = i
                        self.players[i].posession = True
                        self.ball_state = 'DRIBBLING'
                        self.state[-5:-3] = self.state[self.player_posession*4:self.player_posession*4+2]
                        self.state[-3:-1] = self.state[self.player_posession*4+2:self.player_posession*4+4]
                        # if intercepted by defense then turnover
                        if self.player_posession >= 5:
                            print('possession wrong')
                            self.ball_state = 'STOLEN'
                            self.all_actions['stolen'] += 1
                            return False
                        break
                # out of bounds turnover
                if self.state[-5] > self.court_dims[0] - self.force_field:
                    print('OB1')
                    self.ball_state = 'OB'
                    self.all_actions['outofbounds'] += 1
                    return False
                elif self.state[-5] < self.force_field:
                    self.ball_state = 'OB'
                    print('OB2')
                    self.all_actions['outofbounds'] += 1
                    return False
                if self.state[-4] > self.court_dims[1] - self.force_field:
                    self.ball_state = 'OB'
                    print('OB3')
                    self.all_actions['outofbounds'] += 1
                    return False
                elif self.state[-4] < self.force_field:
                    self.ball_state = 'OB'
                    print('OB4')
                    self.all_actions['outofbounds'] += 1
                    return False
            elif (self.ball_state == 'SHOOTING' or 'MIDAIR SHOT'): # shooting
                self.ball_state = 'MIDAIR SHOT'
                distance = np.linalg.norm(self.state[-5:-3] - self.basket_location)
                if distance < self.ball_radius:
                    # % chance making shot based on distance, capped at 30% and 70%
                    shot_chance = 0.3 if self.shot_dist >= 24 else (0.7 if self.shot_dist <= 5 else 0.3 + 0.021*(self.shot_dist-5))
                    if random.choices([0, 1], weights=(1-shot_chance, shot_chance))[0] == 0:
                        self.ball_state = 'MISSED SHOT'
                        self.all_actions['missed_shot'] += 1
                        return False
                    else:
                        self.ball_state = 'MADE SHOT'
                        self.all_actions['made_shot'] += 1
                        return False

    def shift_perspective(self, player_id):
        player_state = deepcopy(self.state)
        if player_id >= 5:
            tmp = player_state[0:20]
            player_state[0:20] = player_state[20:40]
            player_state[20:40] = tmp
            tmp = player_state[-15:-10]
            player_state[-15:-10] = player_state[-10:-5]
            player_state[-10:-5] = tmp
            tmp = player_state[(player_id-5)*4: (player_id-5)*4+4]
            player_state[(player_id-5)*4: (player_id-5)*4+4] = player_state[0:4]
            player_state[0:4] = tmp
        else:
            tmp = player_state[player_id*4:player_id*4+4]
            player_state[player_id*4:player_id*4+4] = player_state[0:4]
            player_state[0:4] = tmp
            tmp = player_state[-15+player_id]
            player_state[-15+player_id] = player_state[-15]
            player_state[-15] = tmp
        return player_state

    def step(self, actions, save_state=False):
        '''
            actions: np.array of 10 actions, one for each player
        '''
        done = False
        prev_player_posession = self.player_posession
        # update player positions
        self.movement(actions)
        print(self.ball_state,  'in step')

        #print(self.player_posession)
        if self.ball_state != 'MIDAIR SHOT' and self.ball_state != 'MIDAIR PASS':
            print('someone has possession')
            # if player has ball posession then check for update in ball state
            b_space = self.players[self.player_posession].ball_handler_space
            # if agent chose to pass to themselves then they're dribbling
            if np.argmax(b_space) == self.player_posession:
                #print('DRIBBLING IN STEP')
                self.ball_state = 'DRIBBLING'
            else:
                # if pass to rim, then shoot, determine if blocked or not, then set ball on its course
                if np.argmax(b_space) == 5:
                    self.ball_state = 'SHOOTING'
                    print('Shooting')
                    self.shot_dist = math.dist(self.state[-5:-3], self.basket_location)
                    self.state[self.player_posession - 15] = 0
                    # check if any defense is blocking
                    for i in range(5, 10):
                        distance = np.linalg.norm(self.state[i*4:i*4+2] - self.state[-5:-3])
                        # probability of blocking based on distance away with max distance for chance to block is 2*ball radius + player radius
                        # highest probability capped at 0.25 if you're within force field
                        if distance < self.player_radius + self.ball_radius:
                            weight = distance * (0.25/(-2*self.ball_radius)) + -(self.player_radius + 2*self.ball_radius)*(0.25/(-2*self.ball_radius))
                            if random.choices([1,2], weights=[weight, 1-weight])[0] == 1:
                                self.ball_state = 'BLOCKED'
                                self.all_actions['blocked'] += 1
                                done = True
                    # rim - ball position norm * ball speed gets ball vector
                    self.state[-3:-1] = (list(self.basket_location) - self.state[-5:-3])/np.linalg.norm(list(self.basket_location) - self.state[-5:-3]) * self.ball_speed
                # if passing, find target player, set ball on its course
                else:
                    self.ball_state = 'PASSING'
                    #print('PASSING IN STEP')
                    # curr target player pos - ball position norm * ball speed gets ball vector
                    target_pos = self.state[np.argmax(b_space)*4:np.argmax(b_space)*4+2]
                    try:
                        self.state[-3:-1] = (target_pos - self.state[-5:-3])/np.linalg.norm(target_pos - self.state[-5:-3]) * self.ball_speed
                    except:
                        pass
                # Remove posession from player
                self.players[self.player_posession].ball_handler_space = np.zeros(6)
                self.players[self.player_posession].posession = False
                
                #self.player_posession = None

        # update ball position
        ball_motion = self.ball_movement()
        if ball_motion == False:
            # if ball_motion == True track it so reward can be adjusted - just keeping it to only false for now
            print('ball not moving')
            done = True
        self.state[-1] -= self.time_increment
        if self.state[-1] <= 0:
            self.all_actions['no_shot'] += 1
            done = True
        if save_state:
            self.all_states.append(deepcopy(self.state))

        onball_rewards = 0
        offball_rewards = []
        defense_rewards = []
        for p in self.players:
            if p.player == prev_player_posession:
                onball_rewards = self.reward_offense_onball(p.player)
            elif p.team == 0:
                offball_rewards.append(self.reward_offense_offball(p.player))
            else:
                defense_rewards.append(self.reward_defense(p.player))

        return onball_rewards, offball_rewards, defense_rewards, done

    def init(self):
        for patch in self.player_circles:
            patch.set_visible(False)
        return tuple()

    def render(self, save_path):
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(0,
                            47),
                      ylim=(0,
                            50))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        

        
        # ax.annotate(24.0, xy=[23.5, 48],
        #                          color='black', horizontalalignment='center',
        #                            verticalalignment='center')

        offense_player_coords = self.all_states[0][:20]
        defense_player_coords = self.all_states[0][20:40]
        ball_coords = self.all_states[0][-5:-3]
        self.player_circles = []
        for i in range(5):
            self.player_circles.append(plt.Circle((offense_player_coords[4*i], offense_player_coords[4*i+1]), radius = self.player_radius/3, color = 'g' ))
            #ax.annotate(i, xy=[offense_player_coords[4*i], offense_player_coords[4*i+1]], color = 'k')
        for i in range(5):
            self.player_circles.append(plt.Circle((defense_player_coords[4*i], defense_player_coords[4*i+1]), radius = self.player_radius/3, color = 'b' ))
            #ax.annotate(i+5, xy=[defense_player_coords[4*i], defense_player_coords[4*i+1]], color = 'k')
        # ball patch        
        self.player_circles.append(plt.Circle((ball_coords[0], ball_coords[1]), radius = self.ball_radius/3, color = 'orange' ))

        for circle in self.player_circles:
                ax.add_patch(circle)
        anim = animation.FuncAnimation(
                         fig, self.animate, init_func=self.init, frames = len(self.all_states),fargs = (ax,), 
                         blit = True, interval=10, repeat = True)
        court = plt.imread("halfcourt.png")
        plt.imshow(court, zorder=0, extent=[0, 47,
                                            50, 0])
        # plt.show()
        f = os.path.join(save_path, "animation.gif")
        writergif = animation.PillowWriter(fps=12) 
        anim.save(f, writer=writergif)
    
        
    def animate(self, frame, ax):
            offense_player_coords = self.all_states[frame][:20]
            defense_player_coords = self.all_states[frame][20:40]
            ball_coords = self.all_states[frame][-5:-3]
            for i in range(5):
                self.player_circles[i].set_visible(True)
                self.player_circles[i].center = offense_player_coords[i*4], offense_player_coords[i*4+1]
                #ax.annotate(i, xy=[offense_player_coords[4*i], offense_player_coords[4*i+1]], color = 'k')
            for i in range(5):
                self.player_circles[i+5].set_visible(True)
                self.player_circles[i+5].center = defense_player_coords[4*i], defense_player_coords[4*i+1]
                #ax.annotate(i+5, xy=[defense_player_coords[4*i], defense_player_coords[4*i+1]], color = 'k')
            # ball circle
            self.player_circles[-1].set_visible(True)
            self.player_circles[-1].center = ball_coords[0], ball_coords[1]
            if frame == 0:
                self.clock_render = ax.text(23.5,48,str(self.all_states[frame][-1]),
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')
            else:
                self.clock_render.set_text(str(self.all_states[frame][-1]))
            
            return self.player_circles
    def close(self):
        pass
    