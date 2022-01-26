# coding:utf-8

number_of_worlds = 10   #Number of mouses that will learn together (in case of federated learning)

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
MAX_AGE = 20000
MEAN_INTERVAL = int(MAX_AGE/100)   #Size of the mean convolutional filter

learning_mode_index = 2
learning_modes = ['Tabular_QLearning', 'Deep_QLearning', 'Federated_Deep_QLearning', 'Federated_SMPC_Deep_QLearning']
assert learning_mode_index <= len(learning_modes)
if learning_mode_index>=2:
    assert number_of_worlds > 1
LEARNING_MODE = learning_modes[learning_mode_index]

nb_maj_required_to_update_main_model= 10 #ou 1000 On MàJ le main model par federative learning tous les x mouvements


# ------Reward and Punishment----
EATEN_BY_CAT = -100
MOVE_REWARD = 1     #Récompense pour avoir fait un mouvement sans être mangée + permet de la pousser a bouger/explorer des états lorsque le chat n'est pas là
#TODO : reward if hits the wall ?
HIT_WALL = -10000    #Mouse can't go on a wall (if we wan't to delete the effect of the wall negative reward : just set HIT_WALL = MOVE_REWARD
TIME_TO_SURVIVE=100     #Durée d'un épisode pour que souris gagne


# determine how many directions can agent moves.
directions = 8   # you may change it to 4: up,down,left and right.


# ------Display----
show_mouses_individual_performances = False
show_variance = True

