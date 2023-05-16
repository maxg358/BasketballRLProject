from nba_simulation import *
from train import *


def random_run(save_path):
  game = NBAGymEnv()
  state_array = []
  count = 0
  while True:
    count+=1
    actions = [[0,0], [0,0],[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
    
    for i, agent in enumerate(game.players):
        actions[i] = agent.act(actions[i])
    _, _, _, done = game.step(actions)
    
    if done:
      break
  game.render(save_path)

def model_run(ball_model_path, offense_model, defense_model, save_path):
  game = NBAGymEnv(random=False)
  movement_space = 2
  ball_handler_space = 6
  num_inputs = 55
  hidden_size = 64
  
  offense_model = AdvantageActorCritic(num_inputs, movement_space, hidden_size, continuous=True)
  defense_model = AdvantageActorCritic(num_inputs, movement_space, hidden_size, continuous=True)
  ball_model = AdvantageActorCritic(num_inputs, ball_handler_space, hidden_size)
  
