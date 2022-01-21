# coding:utf-8

# -----World Setting------
graphic_file = 'resources/world.txt'
grid_width = 50   # pixels of a single grid
wall_color = '#D3CAB4'
cat_color = '#000000'
mouse_color = '#06B1C8'
speed = 9000   # animal speed is 10m/s, the max value supposed to be less than
# 1000.
#TODO : SPEED ?


# -----Learning Parameters---
alpha = 0.1    # learning rate
gamma = 0.9    # importance of next action
epsilon = 0.1  # exploration chance


# ------Reward and Punishment----
EATEN_BY_CAT = -1
MOVE_REWARD = 0 #+1 L'UN OU L'AUTRE JE DIRAIS

# determine how many directions can agent moves.
directions = 8   # you may change it to 4: up,down,left and right.

