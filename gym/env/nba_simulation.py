import math
import warnings
import numpy as np
import gym
import random
from math import sqrt
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib import animation
from utils.data import *
import sys
sys.path.append('..')
from ..utils import *
from ..vis import *

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

    
    def act(self, prev_vector, min_val=-2, max_val=2, decimal_places=1):
        float1 = round(random.uniform(min_val, max_val), decimal_places)
        float2 = round(random.uniform(min_val, max_val), decimal_places)
        action = np.array([float1, float2])
        action /= np.linalg.norm(action)
        action += prev_vector
        action /= np.linalg.norm(action)
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
        self.force_field = 2.5
        self.player_radius = 3.25
        self.ball_radius = 1
        self.player_speed = 0.25 #seconds per unit
        self.basket_location = (0, 25)
        self.court_dims = (47, 50)
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
        self.state = np.array([*r_p[0], 0.0, 0.0, *r_p[1], 0.0, *r_p[2], 0.0, 0.0,
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
            actions: List of 10 actions, one for each player
        '''
        done = False
        return done
       
    def render(self, mode='human'):
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                      ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        start_moment = self.moments[0]
        player_dict = self.player_ids_dict

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')

        annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]

        # Prepare table
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)
        
        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]
        
        home_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[:5]]
        guest_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))

        table = plt.table(cellText=players_data,
                              colLabels=column_labels,
                              colColours=column_colours,
                              colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
                              loc='bottom',
                              cellColours=cell_colours,
                              fontsize=Constant.FONTSIZE,
                              cellLoc='center')
        table.scale(1, Constant.SCALE)
        table_cells = table.properties()['child_artists']
        for cell in table_cells:
            cell._text.set_color('white')

        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                 color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
                         fig, self.update_radius,
                         fargs=(player_circles, ball_circle, annotations, clock_info),
                         frames=len(self.moments), interval=Constant.INTERVAL)
        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])
        plt.show()

    
    def close(self):
        pass
    