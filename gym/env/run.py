from nba_simulation import *

game = NBAGymEnv()
state_array = []
count = 0
while True:
  count+=1
  actions = [[0,0], [0,0],[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
  state_array.append(game.state)
  
  for i, agent in enumerate(game.players):
      actions[-1-i] = agent.act(actions[-1-i])
  done = game.step(actions)
  
  if done or count >2000:
     break
print(state_array[0])
print(state_array[1])
print(state_array[2])
game.render(state_array)
      