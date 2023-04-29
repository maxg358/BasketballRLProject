import math
import warnings
import numpy as np
import gym
import random
from math import sqrt
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
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
        self.movement_space = np.array([0.0, 0.0]) #TODO 
        self.posession = posession
        self.ball_handler_space = None #TODO
        self.team = team

class RandomAgent(Agent):
    def __init__(self, id, posession=False, team=0):
        super(RandomAgent, self).__init__(id, posession)
        self.movement_space = np.array([0.0, 0.0])
        self.max_player_speed = 2
    
    def act(self, prev_vector=[0., 0.], min_val=-2, max_val=2, decimal_places=1):
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
    def __init__(self):
        super(NBAGymEnv, self).__init__()
        '''
            State: dimensions 0-39 are [2D position, 2D direction] + 11D posession
            + 2D ball position + 1D shot clock
            Action: 2D movement for all players except ball handler

        '''
        # Define action and observation spaces
        r_p = self.generate_points()

        self.state = np.array([*r_p[0], 0.0, 0.0, *r_p[1], 0.0, *r_p[2], 0.0, 0.0,
                               *r_p[3], 0.0, 0.0, *r_p[4], 0.0, 0.0, *r_p[5], 0.0, 0.0,
                                *r_p[6], 0.0, 0.0, *r_p[7], 0.0, 0.0, *r_p[8], 0.0, 0.0,
                                 *r_p[9], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 24.0])
        self.all_states = [np.copy(self.state)]
        self.force_field = 2.5
        self.player_radius = 3.25
        self.ball_radius = 1
        self.time_increment = 0.25 #seconds per unit
        self.max_player_speed = 2
        self.basket_location = (0, 25)
        self.court_dims = (47, 50)
        self.all_states = [deepcopy(self.state)]
        self.players = [RandomAgent(0, True), RandomAgent(1), RandomAgent(2), RandomAgent(3), RandomAgent(4), RandomAgent(5, team=1), RandomAgent(6, team=1), RandomAgent(7, team=1), RandomAgent(8, team=1), RandomAgent(9, team=1)]
        # Initialize the play

    def generate_points(self, min_distance=2.5, num_points=10, box_width=47, box_height=50):
        def distance(p1, p2):
            return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def is_valid_point(new_point, points, min_distance):
            for point in points:
                if distance(new_point, point) < min_distance:
                    return False
            return True

        points = []
        attempts = 0

        while len(points) < num_points and attempts < 1000:
            new_point = (random.uniform(0, box_width), random.uniform(0, box_height))

            if is_valid_point(new_point, points, min_distance):
                points.append(new_point)
            else:
                attempts += 1

        if attempts == 1000:
            print("Failed to generate the required number of points with the specified minimum distance.")
            return []

        return points

    def reset(self):
        r_p = self.generate_points()
        self.state = np.array([*r_p[0], 0.0, 0.0, *r_p[1], 0.0, 0.0, *r_p[2], 0.0, 0.0,
                                *r_p[3], 0.0, 0.0, *r_p[4], 0.0, 0.0, *r_p[5], 0.0, 0.0,
                                *r_p[6], 0.0, 0.0, *r_p[7], 0.0, 0.0, *r_p[8], 0.0, 0.0,
                                *r_p[9], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 24.0])

    def reward(self, offense, agent):
        if offense:
            pass
        else:
            pass

    def step(self, actions):
        '''
            actions: np.array of 10 actions, one for each player
        '''
        done = False
        # update player positions - get proposed positions for all and do out of bounds check,
        proposed_pos = np.zeros((10, 2))
        for i, act in enumerate(actions):
            # cap on movement speed
            self.state[i*4+2:i*4+4] += act
            if np.linalg.norm(self.state[i*4+2:i*4+4]) > self.max_player_speed:
                self.state[i*4+2:i*4+4] /= np.linalg.norm(self.state[i*4+2:i*4+4])
                self.state[i*4+2:i*4+4] *= self.max_player_speed
            proposed_pos[i] = self.state[i*4:i*4+2] + self.state[i*4+2:i*4+4]
            # out of bounds check
            if proposed_pos[i][0] > self.court_dims[0]:
                proposed_pos[i][0] = self.court_dims[0]
            elif proposed_pos[i][0] < 0:
                proposed_pos[i][0] = 0
            if proposed_pos[i][1] > self.court_dims[1]:
                proposed_pos[i][1] = self.court_dims[1]
            elif proposed_pos[i][1] < 0:
                proposed_pos[i][1] = 0
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
            self.state[i*4:i*4+2] = proposed_pos[i]
        self.all_states.append(deepcopy(self.state))
        return done
       
    def render(self, state_array):
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(0,
                            47),
                      ylim=(0,
                            50))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        

        clock_info = ax.annotate('', xy=[23.5, 45],
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')

        offense_player_coords = self.all_states[0][:10]
        defense_player_coords = self.all_states[0][10:20]
        player_circles = []
        for i in range(5):
            player_circles.append(plt.Circle((offense_player_coords[i], offense_player_coords[i+1]), radius = self.player_radius, color = 'g' ))
        for i in range(5):
            player_circles.append(plt.Circle((defense_player_coords[2*i], defense_player_coords[2*i+1]), radius = self.player_radius, color = 'b' ))

        # for circle in player_circles:
        #     ax.add_patch(circle)
        
        
        
        # ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
        #                          color=start_moment.ball.color)
        
        # ax.add_patch(ball_circle)
        
        anim = animation.FuncAnimation(
                         fig, self.animate, frames = len(state_array),fargs = (ax,), blit = True,
                         
                          interval=10)
        court = plt.imread("halfcourt.png")
        plt.imshow(court, zorder=0, extent=[0, 47,
                                            50, 0])
        plt.show()
    
        
    def animate(self, frame, ax):
            offense_player_coords = self.all_states[frame][:10]
            defense_player_coords = self.all_states[frame][10:20]
            player_circles = []
            for i in range(5):
                player_circles.append(plt.Circle((offense_player_coords[i], offense_player_coords[i+1]), radius = self.player_radius/2, color = 'g' ))
            for i in range(5):
                player_circles.append(plt.Circle((defense_player_coords[2*i], defense_player_coords[2*i+1]), radius = self.player_radius/2, color = 'b' ))

            for circle in player_circles:
                ax.add_patch(circle)
            return player_circles
    def close(self):
        pass
    