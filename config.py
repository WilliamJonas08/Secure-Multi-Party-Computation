# coding:utf-8

number_of_worlds = 2    #Number of mouses that will learn together (in case of federated learning)

# -----World Setting------
graphic_file = 'resources/world.txt'
grid_width = 50   # pixels of a single grid
wall_color = '#D3CAB4'
cat_color = '#000000'
mouse_color = '#06B1C8'
speed = 1000   # animal speed is 10m/s, the max value supposed to be less than



# -----Learning Parameters---
alpha = 0.1    # learning rate
gamma = 0.9    # importance of next action
epsilon = 0.1  # exploration chance
MAX_AGE = 1000
MEAN_INTERVAL = 10    #int(MAX_AGE/1000)

# ------Reward and Punishment----
EATEN_BY_CAT = -100
MOVE_REWARD = +1 #Récompense pour avoir fait un mouvement sans être mangée
TIME_TO_SURVIVE=100 #Durée d'un épisode pour que souris gagne

# determine how many directions can agent moves.
directions = 8   # you may change it to 4: up,down,left and right.

