# coding:utf-8

number_of_worlds = 1   #Number of mouses that will learn together (in case of federated learning)

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
MEAN_INTERVAL = int(MAX_AGE/50)   #Size of the mean convolutional filter

learning_modes = ['Tabular Q-Learning', 'Deep Q-Learning']
learning_mode_index = 0
assert learning_mode_index <= len(learning_modes)
LEARNING_MODE = learning_modes[learning_mode_index]


update_of_main_model= 10 #ou 1000 On MàJ le main model par federative learning tous les x mouvements

# ------Reward and Punishment----
EATEN_BY_CAT = -100
MOVE_REWARD = 0 #Récompense pour avoir fait un mouvement sans être mangée
TIME_TO_SURVIVE=100 #Durée d'un épisode pour que souris gagne


# determine how many directions can agent moves.
directions = 8   # you may change it to 4: up,down,left and right.


# ------Display----
show_mouses_individual_performances = False
show_variance = False

