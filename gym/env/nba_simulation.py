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
        self.movement_space = np.array([0.0, 0.0])
        self.posession = posession
        #TODO: Dribble (pass to yourself), Pass to 4 others, Shoot (a pass to the rim)
        self.ball_handler_space = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        self.team = team

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
    def __init__(self):
        super(NBAGymEnv, self).__init__()
        '''
            State: dimensions 0-39 are [2D position, 2D direction] + 11D posession
            + 2D ball position + 1D shot clock + 5D to say which player you are
            Action: 2D movement for all players except ball handler

        '''
        # Define action and observation spaces
        r_p = self.generate_points()
        # 10 players’ info ( 2D position vectors + 2D direction vector) + 11D who has the ball + 2D ball position + 2D ball vector + 1D shot clock
        self.state = np.array([*r_p[0], 0.0, 0.0, *r_p[1], 0.0, 0.0, *r_p[2], 0.0, 0.0,
                                *r_p[3], 0.0, 0.0, *r_p[4], 0.0, 0.0, *r_p[5], 0.0, 0.0,
                                *r_p[6], 0.0, 0.0, *r_p[7], 0.0, 0.0, *r_p[8], 0.0, 0.0,
                                *r_p[9], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, *r_p[0], 0.0, 0.0, 24.0])
        self.player_posession = 0
        self.all_states = [np.copy(self.state)]
        self.force_field = 3.25
        self.player_radius = 3.25
        self.ball_radius = 1.5
        self.time_increment = 0.25 #seconds per unit
        self.max_player_speed = .5
        self.ball_speed = 1
        self.ball_state = 'DRIBBLING' # or 'PASSING' or 'SHOOTING' of 'MIDAIR'
        self.basket_location = (6, 25)
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
        r_p = self.generate_points()
        self.state = np.array([*r_p[0], 0.0, 0.0, *r_p[1], 0.0, 0.0, *r_p[2], 0.0, 0.0,
                                *r_p[3], 0.0, 0.0, *r_p[4], 0.0, 0.0, *r_p[5], 0.0, 0.0,
                                *r_p[6], 0.0, 0.0, *r_p[7], 0.0, 0.0, *r_p[8], 0.0, 0.0,
                                *r_p[9], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, *r_p[0], 0.0, 0.0, 24.0])

    def reward(self):
        pass


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
            print('dribbling')
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
                        # if near defense, give 10% chance of stealing, otherwise let it go through
                        if i >= 5 and random.choice([1,2,3,4,5,6,7,8,9,10]) != 10 :
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
                            return False
                        break
                # out of bounds turnover
                if self.state[-5] > self.court_dims[0] - self.force_field:
                    print('OB1')
                    return False
                elif self.state[-5] < self.force_field:
                    print('OB2')
                    return False
                if self.state[-4] > self.court_dims[1] - self.force_field:
                    print('OB3')
                    return False
                elif self.state[-4] < self.force_field:
                    print('OB4')
                    return False
            elif (self.ball_state == 'SHOOTING' or 'MIDAIR SHOT'): # shooting
                self.ball_state = 'MIDAIR SHOT'
                distance = np.linalg.norm(self.state[-5:-3] - self.basket_location)
                if distance < self.ball_radius:
                    # 30% chance missing
                    if random.choice([1,2,3,4,5,6,7,8,9,10]) > 7:
                        print('Missed Shot')
                        return False
                    # 70% chance making
                    else:
                        print('Made shot')
                        return False

    def step(self, actions):
        '''
            actions: np.array of 10 actions, one for each player
        '''
        done = False
        # update player positions
        self.movement(actions)
        print(self.ball_state,  'in step')

        #print(self.player_posession)
        if self.ball_state != 'MIDAIR SHOT'and self.ball_state != 'MIDAIR PASS':
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
                    #print("SHOOTING IN  STEP")
                    # check if any defense is blocking
                    for i in range(5):
                        distance = np.linalg.norm(self.state[(i+5)*4:(i+5)*4+2] - self.state[-5:-3])
                        # probability of blocking based on distance away with max distance for chance to block is 2*ball radius + player radius
                        # highest probability capped at 0.25 if you're within force field
                        if distance < self.player_radius + 2*self.ball_radius:
                            weight = distance * (0.25/(-2*self.ball_radius)) + -(self.player_radius + 2*self.ball_radius)*(0.25/(-2*self.ball_radius))
                            if random.choices([1,2], weights=[weight, 1-weight]) == 1:
                                done = False
                    # rim - ball position norm * ball speed gets ball vector
                    self.state[-3:-1] = (list(self.basket_location) - self.state[-5:-3])/np.linalg.norm(list(self.basket_location) - self.state[-5:-3]) * self.ball_speed
                # if passing, find target player, set ball on its course
                else:
                    self.ball_state = 'PASSING'
                    #print('PASSING IN STEP')
                    # curr target player pos - ball position norm * ball speed gets ball vector
                    target_pos = self.state[np.argmax(b_space)*4:np.argmax(b_space)*4+2]
                    self.state[-3:-1] = (target_pos - self.state[-5:-3])/np.linalg.norm(target_pos - self.state[-5:-3]) * self.ball_speed
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
            done = True
        self.all_states.append(deepcopy(self.state))
        #print(done)
        return done
       
    def render(self):
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
        player_circles = []
        for i in range(5):
            player_circles.append(plt.Circle((offense_player_coords[4*i], offense_player_coords[4*i+1]), radius = self.player_radius, color = 'g' ))
            #ax.annotate(i, xy=[offense_player_coords[4*i], offense_player_coords[4*i+1]], color = 'k')
        for i in range(5):
            player_circles.append(plt.Circle((defense_player_coords[4*i], defense_player_coords[4*i+1]), radius = self.player_radius, color = 'b' ))
            #ax.annotate(i+5, xy=[defense_player_coords[4*i], defense_player_coords[4*i+1]], color = 'k')
        
        player_circles.append(plt.Circle((ball_coords[0], ball_coords[1]), radius = self.ball_radius/3, color = 'orange' ))
        # for circle in player_circles:
        #     ax.add_patch(circle)
        # ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
        #                          color=start_moment.ball.color)
        # ax.add_patch(ball_circle)
        
        anim = animation.FuncAnimation(
                         fig, self.animate, frames = len(self.all_states),fargs = (ax,), 
                         blit = True, interval=10,repeat = True)
        court = plt.imread("halfcourt.png")
        plt.imshow(court, zorder=0, extent=[0, 47,
                                            50, 0])
        plt.show()
    
        
    def animate(self, frame, ax):
            offense_player_coords = self.all_states[frame][:20]
            defense_player_coords = self.all_states[frame][20:40]
            ball_coords = self.all_states[frame][-5:-3]
            player_circles = []
            for i in range(5):
                player_circles.append(plt.Circle((offense_player_coords[i*4], offense_player_coords[i*4+1]), radius = self.player_radius/3, color = 'g' ))
                #ax.annotate(i, xy=[offense_player_coords[4*i], offense_player_coords[4*i+1]], color = 'k')
            for i in range(5):
                player_circles.append(plt.Circle((defense_player_coords[4*i], defense_player_coords[4*i+1]), radius = self.player_radius/3, color = 'b' ))
                #ax.annotate(i+5, xy=[defense_player_coords[4*i], defense_player_coords[4*i+1]], color = 'k')
            # ball circle
            player_circles.append(plt.Circle((ball_coords[0], ball_coords[1]), radius = self.ball_radius/3, color = 'orange' ))
            for circle in player_circles:
                
                ax.add_patch(circle)
            ax.text(23.5,48,str(self.all_states[frame][-1]),
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')
            
            
            return player_circles
    def close(self):
        pass
    