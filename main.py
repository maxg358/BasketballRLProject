import py7zr
import pandas as pd
import os
import pickle

def main():
    # directory = r'data\2016.NBA.Raw.SportVU.Game.Logs'
    # for filename in os.listdir(directory):
    #     f = os.path.join(directory, filename)
    #     with py7zr.SevenZipFile(f, mode='r') as z:
    #         z.extractall(path = r"extracted_games")
    #     print('success')
    directory = r'extracted_games'
    counter = 0
    for filename in os.listdir(directory):
        

        f = os.path.join(directory, filename)
        game = pd.read_json(f)
        print(game['events'][3]['moments'][18])
        return
        # game_state_vectors = []
        
        # visiting_players = game['events'][0]['visitor']['players']
        # home_players = game['events'][0]['home']['players']
        # player_dict = {} # mapping player id to position, can also add team and whatever other features we need
        # player_position_dict = {}
        # for entry in visiting_players:
        #     player_dict[entry['playerid']] = entry['position']
        #     player_position_dict[entry['playerid']] = [0,0]
        # for entry in home_players:
        #     player_dict[entry['playerid']] = entry['position']
        #     player_position_dict[entry['playerid']] = [0,0]
        # player_position_dict[0] = [0,0]
        
        

        
        
        # ball_position = [0,0,0]
       
        # for event in game['events']:
            
        #     for moment in event['moments']:
        #         state_vector = []

        #         ball = moment[5][0]
                
        #         players = moment[5][1:]
        #         for player in players: # 60 of the state vector entries
        #             state_vector.append(player[1]) #id
        #             state_vector.append(player[2]) #x position
        #             state_vector.append(player[3]) #y position
        #             try:
        #                 state_vector.append(player_dict[player[1]])# player position (G/F/C)
        #             except:
        #                 state_vector.append('G')
        #                 #If player is on the court, append motion
                    
        #             state_vector.append(round(player[2] - player_position_dict[player[1]][0], 4) ) #player directions
        #             state_vector.append(round(player[3] - player_position_dict[player[1]][1], 4))
                    

                        
                    
        #             player_position_dict[player[1]] = [player[2], player[3]]
                
        #         state_vector.append(ball[2]) # ball x coordinate
        #         state_vector.append(ball[3]) # ball y coordinate
        #         state_vector.append(ball[4]) # ball radius (proxy for z coordinate - unsure how else to get it idk)
        #         state_vector.append(round(ball[2] - ball_position[0], 4)) #ball direction x
        #         state_vector.append(round(ball[3] - ball_position[1], 4)) #ball direction y
        #         state_vector.append(round(ball[4] - ball_position[2], 4)) #ball direction z
        #         ball_position = [ball[2], ball[3], ball[4]]
        #         state_vector.append(moment[0]) # quarter
        #         state_vector.append(moment[2]) # game clock
        #         state_vector.append(moment[3]) # shot clock
                
                
        #     game_state_vectors.append(state_vector)
        #     pickle_out = open("game_states/game" + str(counter) + ".pickle","wb")
        #     pickle.dump(game_state_vectors, pickle_out)
        
            
                     



                
        # game_state_vectors.append(state_vector) 
        # print('game done')
    
        

if __name__ == "__main__":
    main()