from nba_simulation import *

game = NBAGymEnv()
while True:
  actions = []
  for agent in game.players:
      actions.append(agent.act())
  done = game.step(actions)
  game.render()
  if done:
     break
      