# coding:utf-8

import random
import setup
import qlearn
import config as cfg
from queue import Queue
import numpy as np
# reload(setup) 
# reload(qlearn)



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

def moving_average(x, window_size):
    #Modes : 'valid', 'same', 'full'
    return np.convolve(x, np.ones(window_size), 'valid') / window_size

def compute_mean_var_performance_over_mouses(mouses_performances):
    nb_mouses = cfg.number_of_worlds

    performances_at_each_age = np.empty((nb_mouses,cfg.MAX_AGE))
    for mouse_id in range(nb_mouses):
        age = 0
        for survival_time in mouses_performances[mouse_id]:
            for foo in range(survival_time):
                performances_at_each_age[mouse_id][age] = survival_time
                age+=1

    min_mouse_ages = np.min([sum(mouses_performances[i]) for i in range(nb_mouses)])   #All mouses doesn't have the time to register the same number of ages
    avg_mouse_lifetime = int(np.mean([np.mean(perfs) for perfs in mouses_performances]))

    performances_at_each_age = performances_at_each_age[:,:min_mouse_ages]
    mean_performances = np.array([np.mean(performances_at_each_age[:,age]) for age in range(min_mouse_ages)])
    var_performances = np.array([np.var(performances_at_each_age[:,age]) for age in range(min_mouse_ages)])

    #Display purpose only (hide irregularities due to unfinished last iterations)
    mean_performances[:min_mouse_ages-avg_mouse_lifetime]
    var_performances[:min_mouse_ages-avg_mouse_lifetime]

    #Take moving average of values
    mean_performances  = moving_average(mean_performances, window_size=cfg.MEAN_INTERVAL)
    var_performances  = moving_average(var_performances, window_size=cfg.MEAN_INTERVAL)

    return mean_performances, var_performances


#if __name__ == '__main__':

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


#Get performances through ages
performances = []
for world in worlds:
    performance = world.get_mouse_performance()
    performances.append(performance)

mean_performance, var_performances = compute_mean_var_performance_over_mouses(mouses_performances=performances)
avg_mouse_lifetime = int(np.mean([np.mean(perfs) for perfs in performances]))
#print("SUM PERF", [np.sum(perf) for perf in performances])
#print("PERFORMANCE MOUSE 0", performances[0])
#print("MEAN PERFORMANCES", mean_performance)


#Plot
import matplotlib.pyplot as plt

plt.figure(1)
plt.title("Mouses lifetime")
if cfg.show_mouses_individual_performances:
    #Creating graphs
    graphs = [[] for i in range(nb_worlds)]
    for age in range(age_max-avg_mouse_lifetime):
        for world_id in range(nb_worlds):
            mean = mean_average(performances[world_id], age)
            graphs[world_id].append(mean)

    #Plot graphs
    for world_id in range(nb_worlds):
        plt.plot(graphs[world_id], linestyle='--', label=f'Mouse {world_id}')

plt.plot(mean_performance, linewidth=4, label="Mean mouses performance")

if cfg.show_variance :
    plt.figure(2)
    plt.title("Variance of mouses performance")
    plt.plot(var_performances)

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