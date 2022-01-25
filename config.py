# coding:utf-8

# -----World Setting------
graphic_file = 'resources/world.txt'
grid_width = 50   # pixels of a single grid
wall_color = '#D3CAB4'
cat_color = '#000000'
mouse_color = '#06B1C8'
speed = 1000   # animal speed is 10m/s, the max value supposed to be less than
# 1000


# -----Learning Parameters---
alpha = 0.1    # learning rate
gamma = 0.9    # importance of next action
epsilon = 0.7  # exploration chance (decreasing over time)
#TODO : write min eps here
MAX_AGE = 5000

# ------Reward and Punishment----
EATEN_BY_CAT = -10
MOVE_REWARD = +0 #Récompense pour avoir fait un mouvement sans être mangée
HIT_WALL = -1000
TIME_TO_SURVIVE = 100 #Durée d'un épisode pour que souris gagne

# determine how many directions can agent moves.
directions = 8   # you may change it to 4: up,down,left and right.

