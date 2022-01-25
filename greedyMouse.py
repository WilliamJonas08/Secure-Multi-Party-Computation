# coding:utf-8

import random
import setup
import qlearn
import config as cfg
from queue import Queue
import numpy as np
# reload(setup) 
# reload(qlearn)


# def pick_random_location():
#     while 1:
#         x = random.randrange(world.width)
#         y = random.randrange(world.height)
#         cell = world.get_cell(x, y)
#         if not (cell.wall or len(cell.agents) > 0):
#             return cell


# def pick_random_location_bis():
#     while 1:
#         x = random.randrange(world_bis.width)
#         y = random.randrange(world_bis.height)
#         cell = world_bis.get_cell(x, y)
#         if not (cell.wall or len(cell.agents) > 0):
#             return cell


#
# class Cat(setup.Agent):
#     def __init__(self, filename):
#         self.cell = None
#         self.catWin = 0
#         self.color = cfg.cat_color
#         with open(filename) as f:
#             lines = f.readlines()
#         lines = [x.rstrip() for x in lines]
#         self.fh = len(lines) #height plate
#         self.fw = max([len(x) for x in lines]) #width plate
#         self.grid_list = [[1 for x in range(self.fw)] for y in range(self.fh)]
#         self.move = [(0, -1), (1, -1), (
#                 1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
#
#         for y in range(self.fh):
#             line = lines[y]
#             for x in range(min(self.fw, len(line))):
#                 t = 1 if (line[x] == 'X') else 0
#                 self.grid_list[y][x] = t
#
#         print('cat init success......')
#
#     # using BFS algorithm to move quickly to target.
#     def bfs_move(self, target):
#         if self.cell == target:
#             return
#
#         for n in self.cell.neighbors:
#             if n == target:
#                 self.cell = target  # if next move can go towards target
#                 return
#
#         best_move = None
#         q = Queue()
#         start = (self.cell.y, self.cell.x)
#         end = (target.y, target.x)
#         q.put(start)
#         step = 1
#         V = {}
#         preV = {}
#         V[(start[0], start[1])] = 1
#
#         print('begin BFS......')
#         while not q.empty():
#             grid = q.get()
#
#             for i in range(8):
#                 ny, nx = grid[0] + self.move[i][0], grid[1] + self.move[i][1]
#                 if nx < 0 or ny < 0 or nx > (self.fw-1) or ny > (self.fh-1):
#                     continue
#                 if self.get_value(V, (ny, nx)) or self.grid_list[ny][nx] == 1:  # has visit or is wall.
#                     continue
#
#                 preV[(ny, nx)] = self.get_value(V, (grid[0], grid[1]))
#                 if ny == end[0] and nx == end[1]:
#                     V[(ny, nx)] = step + 1
#                     seq = []
#                     last = V[(ny, nx)]
#                     while last > 1:
#                         k = [key for key in V if V[key] == last]
#                         seq.append(k[0])
#                         assert len(k) == 1
#                         last = preV[(k[0][0], k[0][1])]
#                     seq.reverse()
#                     print(seq)
#
#                     best_move = world.grid[seq[0][0]][seq[0][1]]
#
#                 q.put((ny, nx))
#                 step += 1
#                 V[(ny, nx)] = step
#
#         if best_move is not None:
#             self.cell = best_move
#
#         else:
#             dir = random.randrange(cfg.directions)
#             self.go_direction(dir)
#             print("!!!!!!!!!!!!!!!!!!")
#
#     def get_value(self, mdict, key):
#         try:
#             return mdict[key]
#         except KeyError:
#             return 0
#
#     def update(self):
#         print('cat update begin..')
#         if self.cell != mouse.cell:
#             self.bfs_move(mouse.cell)
#             print('cat move..')



# class Cat_bis(setup.Agent):
#     def __init__(self, filename):
#         self.cell = None
#         self.catWin = 0
#         self.color = cfg.cat_color
#         with open(filename) as f:
#             lines = f.readlines()
#         lines = [x.rstrip() for x in lines]
#         self.fh = len(lines) #height plate
#         self.fw = max([len(x) for x in lines]) #width plate
#         self.grid_list = [[1 for x in range(self.fw)] for y in range(self.fh)]
#         self.move = [(0, -1), (1, -1), (
#             1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
#
#         for y in range(self.fh):
#             line = lines[y]
#             for x in range(min(self.fw, len(line))):
#                 t = 1 if (line[x] == 'X') else 0
#                 self.grid_list[y][x] = t
#
#         print('cat init success......')
#
#     # using BFS algorithm to move quickly to target.
#     def bfs_move(self, target):
#         if self.cell == target:
#             return
#
#         for n in self.cell.neighbors:
#             if n == target:
#                 self.cell = target  # if next move can go towards target
#                 return
#
#         best_move = None
#         q = Queue()
#         start = (self.cell.y, self.cell.x)
#         end = (target.y, target.x)
#         q.put(start)
#         step = 1
#         V = {}
#         preV = {}
#         V[(start[0], start[1])] = 1
#
#         print('begin BFS......')
#         while not q.empty():
#             grid = q.get()
#
#             for i in range(8):
#                 ny, nx = grid[0] + self.move[i][0], grid[1] + self.move[i][1]
#                 if nx < 0 or ny < 0 or nx > (self.fw-1) or ny > (self.fh-1):
#                     continue
#                 if self.get_value(V, (ny, nx)) or self.grid_list[ny][nx] == 1:  # has visit or is wall.
#                     continue
#
#                 preV[(ny, nx)] = self.get_value(V, (grid[0], grid[1]))
#                 if ny == end[0] and nx == end[1]:
#                     V[(ny, nx)] = step + 1
#                     seq = []
#                     last = V[(ny, nx)]
#                     while last > 1:
#                         k = [key for key in V if V[key] == last]
#                         seq.append(k[0])
#                         assert len(k) == 1
#                         last = preV[(k[0][0], k[0][1])]
#                     seq.reverse()
#                     print(seq)
#
#                     best_move = world_bis.grid[seq[0][0]][seq[0][1]]
#
#                 q.put((ny, nx))
#                 step += 1
#                 V[(ny, nx)] = step
#
#         if best_move is not None:
#             self.cell = best_move
#
#         else:
#             dir = random.randrange(cfg.directions)
#             self.go_direction(dir)
#             print("!!!!!!!!!!!!!!!!!!")
#
#     def get_value(self, mdict, key):
#         try:
#             return mdict[key]
#         except KeyError:
#             return 0
#
#     def update(self):
#         print('cat update begin..')
#         if self.cell != mouse_bis.cell:
#             self.bfs_move(mouse_bis.cell)
#             print('cat move..')



# class Mouse(setup.Agent):
#     def __init__(self):
#         self.ai = None
#         self.ai = qlearn.QLearn(actions=range(cfg.directions), input_size=8, alpha=0.1, gamma=0.9, epsilon=0.1)
# #TODO self.ai =  contextual bandit agent ?
#         self.catWin = 0
#         self.mouseWin = 0
# #self.mouselife
#         self.lastState = None
#         self.lastAction = None
#         self.color = cfg.mouse_color
#
#         self.iterations=0
#         self.list_iterations=[]
#
#         print('mouse init...')
#
#     def update(self):
#         print('mouse update begin...')
#         state = self.calculate_state()
#         reward = cfg.MOVE_REWARD
#         self.iterations+=1
# #TODO dépend du choix de récompense dans config. !
#
#         if self.cell == cat.cell:
#             print('eaten by cat...')
#             self.catWin += 1
#             reward = cfg.EATEN_BY_CAT
#             if self.lastState is not None:
#                 self.ai.learn(self.lastState, self.lastAction, state, reward, is_last_state=True)
#                 print('mouse learn...')
#             self.lastState = None
#             self.list_iterations.append(self.iterations)
#             self.iterations=0
#             self.cell = pick_random_location()
#             print('mouse random generate..')
#             return
#
#         elif self.iterations>=cfg.TIME_TO_SURVIVE: #On définit le mouseWin
#             self.mouseWin += 1
#             self.lastState = None
#             self.list_iterations.append(self.iterations)
#             self.iterations=0
#             self.cell = pick_random_location()
#             print('mouse random generate..')
#             return
#         #if self.cell == cheese.cell:
#         #    self.mouseWin += 1
#         #    reward = cfg.EAT_CHEESE
#         #    cheese.cell = pick_random_location()
#
#         if self.lastState is not None: #souris non mangée
#             self.ai.learn(self.lastState, self.lastAction, state, reward, is_last_state=False)
#
#         # choose a new action and execute it
#         action = self.ai.choose_action(state)
#         self.lastState = state
#         self.lastAction = action
#         self.go_direction(action)
#
#     def calculate_state(self):
#         """
#         Return : State sous la forme d'un array de valeurs (cell value) correspondant aux cellules adjacentes à la cellule courante de la souris
#         Size array : 8
#         """
#         def cell_value(cell):
#             if cat.cell is not None and (cell.x == cat.cell.x and cell.y == cat.cell.y):
#                 return 3
# #            elif cheese.cell is not None and (cell.x == cheese.cell.x and cell.y == cheese.cell.y):
# #               return 2
#             else:
#                 return 1 if cell.wall else 0
#
#         dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# #TODO : revoir et comprendre
#         return np.array([cell_value(world.get_relative_cell(self.cell.x + dir[0], self.cell.y + dir[1])) for dir in dirs])
#
#     def return_liste_performances(self):
#         return(self.list_iterations)


# class Mouse_bis(setup.Agent):
#     def __init__(self):
#         self.ai = None
#         self.ai = qlearn.QLearn(actions=range(cfg.directions), input_size=8, alpha=0.1, gamma=0.9, epsilon=0.1)
#         #TODO self.ai =  contextual bandit agent ?
#         self.catWin = 0
#         self.mouseWin = 0
#         #self.mouselife
#         self.lastState = None
#         self.lastAction = None
#         self.color = cfg.mouse_color
#
#         self.iterations=0
#         self.list_iterations=[]
#
#         print('mouse init...')
#
#     def update(self):
#         print('mouse update begin...')
#         state = self.calculate_state()
#         reward = cfg.MOVE_REWARD
#         self.iterations+=1
#         #TODO dépend du choix de récompense dans config. !
#
#         if self.cell == cat_bis.cell:
#             print('eaten by cat...')
#             self.catWin += 1
#             reward = cfg.EATEN_BY_CAT
#             if self.lastState is not None:
#                 self.ai.learn(self.lastState, self.lastAction, state, reward, is_last_state=True)
#                 print('mouse learn...')
#             self.lastState = None
#             self.list_iterations.append(self.iterations)
#             self.iterations=0
#             self.cell = pick_random_location_bis()
#             print('mouse random generate..')
#             return
#
#         elif self.iterations>=cfg.TIME_TO_SURVIVE: #On définit le mouseWin
#             self.mouseWin += 1
#             self.lastState = None
#             self.list_iterations.append(self.iterations)
#             self.iterations=0
#             self.cell = pick_random_location()
#             print('mouse random generate..')
#             return
#         #if self.cell == cheese.cell:
#         #    self.mouseWin += 1
#         #    reward = cfg.EAT_CHEESE
#         #    cheese.cell = pick_random_location()
#
#         if self.lastState is not None: #souris non mangée
#             self.ai.learn(self.lastState, self.lastAction, state, reward, is_last_state=False)
#
#         # choose a new action and execute it
#         action = self.ai.choose_action(state)
#         self.lastState = state
#         self.lastAction = action
#         self.go_direction(action)
#
#     def calculate_state(self):
#         """
#         Return : State sous la forme d'un array de valeurs (cell value) correspondant aux cellules adjacentes à la cellule courante de la souris
#         Size array : 8
#         """
#         def cell_value(cell):
#             if cat_bis.cell is not None and (cell.x == cat_bis.cell.x and cell.y == cat_bis.cell.y):
#                 return 3
#             #            elif cheese.cell is not None and (cell.x == cheese.cell.x and cell.y == cheese.cell.y):
#             #               return 2
#             else:
#                 return 1 if cell.wall else 0
#
#         dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
#         #TODO : revoir et comprendre
#         return np.array([cell_value(world_bis.get_relative_cell(self.cell.x + dir[0], self.cell.y + dir[1])) for dir in dirs])
#
#     def return_liste_performances(self):
#         return(self.list_iterations)


##On entraîne en parallèle 2 souris dans 2 mondes séparés

def mean_average(list_perf, age, intervalle=cfg.MEAN_INTERVAL):
    i, val = 0, 0
    Trouve=False
    while i<len(list_perf) and not(Trouve):
        val+=list_perf[i]
        if val>=age:
            Trouve=True
        else:
            i+=1
    if not(Trouve):
        return(Trouve)
    else:
        if i<(intervalle-2):
            return(list_perf[i])
        else:
            mean_avg=0
            for j in range(intervalle):
                mean_avg+=list_perf[i-j]
            mean_avg=mean_avg/intervalle
            return(mean_avg)

#if __name__ == '__main__':
#World 1
#mouse = Mouse()
#cat = Cat(filename='resources/world.txt')
#world = setup.World(filename='resources/world.txt')
#world.add_agent(mouse)
#world.add_agent(cat, cell=pick_random_location())
#world.display.activate()
#world.display.speed = cfg.speed

#World 2
#mouse_bis = Mouse_bis()
#cat_bis = Cat_bis(filename='resources/world_bis.txt') #world_copy.txt
#world_bis = setup.World_bis(filename='resources/world_bis.txt') #setup.World
#world_bis.add_agent(mouse_bis)
#world_bis.add_agent(cat_bis,cell=pick_random_location_bis())
#world_bis.display.activate() ##POSE PB
#world_bis.display.speed = cfg.speed


#Initialisation worlds
nb_worlds = cfg.number_of_worlds
worlds = []
for world_id in range(nb_worlds):
    world = setup.World(filename='resources/world.txt')     #TODO : peut etre créer plusieurs fichiers tkt worlds
    world.add_agents()
    if world_id ==0:
        world.display.activate()
        world.display.speed = cfg.speed

    worlds.append(world)


#Run code
#while 1:
age_max = cfg.MAX_AGE
for i in range(age_max):
    for world in worlds:
        world.update(world.mouse.mouseWin, world.mouse.catWin)
    #world.update(mouse.mouseWin, mouse.catWin)
    #world_bis.update(mouse_bis.mouseWin, mouse_bis.catWin)

#Get performances through ages
performances = []
for world in worlds:
    performance = world.get_mouse_performance()
    performances.append(performance)
#a=mouse.return_liste_performances() #print(a)
#b=mouse_bis.return_liste_performances() #print(b)

#Plot
import matplotlib.pyplot as plt
graphs = [[] for i in range(nb_worlds)]
#my_graph=[]
#my_graph2=[]
for age in range(age_max):
    for world_id in range(nb_worlds):
        mean = mean_average(performances[world_id], age)
        graphs[world_id].append(mean)
    #my_graph.append(mean_average(a, i))
    #my_graph2.append(mean_average(b, i))
for world_id in range(nb_worlds):
    plt.plot(graphs[world_id], label=f'Mouse {world_id}')
#plt.plot(my_graph[:-10], c='b', label='Mouse 1')
#plt.plot(my_graph2[:-10], c='r', label='Mouse 2')
plt.legend(loc='best')
plt.show()

#PB Tkinter car pas possibilités plusieurs instances Tk en parallèle
#Solution envisagée : Tk.TopLevel pour créer fenêtres secondaires
#PB car TopLevel semble dépendre de Tkinter et partage ses données

#New sol : Faire GIF souris seul et souris fédérée au même âge pour comparer comportements
#Afficher moyennes mobiles durée épisodes pour 2

#Pb de longueur de liste itérations selon performances ; comment avoir graphes comparables
#pour des abscisses différentes ? => SOL : Revenir aux âges à partir de listes et faire moy
#pondérées autour de l'âge ?

#Comprendre pourquoi second agent ne fonctionne pas bien

#Problème utilisation souris 2 sans 1 : variables souris 1 utilisées dans déf des différentes classes
#SOL : Créer des classes différentes pour chaque combinaison (world, cat, mouse)