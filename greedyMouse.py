# coding:utf-8

import random
import setup
import qlearn
import config as cfg
import time
import os

import torch

from queue import Queue
import numpy as np
# reload(setup) 
# reload(qlearn)

if cfg.LEARNING_MODE == 'Federated_SMPC_Deep_QLearning':
    #Pour éviter d'importer lorsque non nécessaire
    #!pip install crypten
    import crypten
    crypten.init()

    def encrypted_federative_average(model_dict): #Remplace federative_average
        """
        Computes average of weights updated from all agents through a sMPC method (by default using Crypten encrypted tensors)
        """
        nb_agents=len(model_dict)
        weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision=weights_and_bias_of_model(model_dict["model0"])
        linear1_mean_weight_enc = torch.zeros(size=weight_linear1.shape)
        linear1_mean_bias_enc = torch.zeros(size=bias_linear1.shape)
        linear2_mean_weight_enc = torch.zeros(size=weight_linear2.shape)
        linear2_mean_bias_enc = torch.zeros(size=bias_linear2.shape)
        decision_mean_weight_enc = torch.zeros(size=weight_decision.shape)
        decision_mean_bias_enc = torch.zeros(size=bias_decision.shape)

        with torch.no_grad():
            for i in range(nb_agents):
                weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision = weights_and_bias_of_model(model_dict["model"+str(i)])

                #On encrypte les données du modèle avec Crypten
                weight_linear1_enc, bias_linear1_enc = crypten.cryptensor(weight_linear1), crypten.cryptensor(bias_linear1)
                weight_linear2_enc, bias_linear2_enc = crypten.cryptensor(weight_linear2), crypten.cryptensor(bias_linear2)
                weight_decision_enc, bias_decision_enc = crypten.cryptensor(weight_decision), crypten.cryptensor(bias_decision)

                #On fait la somme sur les données cryptées
                linear1_mean_weight_enc = weight_linear1_enc +  linear1_mean_weight_enc
                linear1_mean_bias_enc = bias_linear1_enc + linear1_mean_bias_enc
                linear2_mean_weight_enc = weight_linear2_enc + linear2_mean_weight_enc
                linear2_mean_bias_enc = bias_linear2_enc + linear2_mean_bias_enc
                decision_mean_weight_enc = weight_decision_enc + decision_mean_weight_enc
                decision_mean_bias_enc = bias_decision_enc + decision_mean_bias_enc

            #On calcule la moyenne sur les données cryptées
            linear1_mean_weight_enc = linear1_mean_weight_enc/nb_agents
            linear1_mean_bias_enc = linear1_mean_bias_enc/nb_agents
            linear2_mean_weight_enc = linear2_mean_weight_enc/nb_agents
            linear2_mean_bias_enc = linear2_mean_bias_enc/nb_agents
            decision_mean_weight_enc = decision_mean_weight_enc/nb_agents
            decision_mean_bias_enc = decision_mean_bias_enc/nb_agents

            #On décrypte les moyennes
            linear1_mean_weight = linear1_mean_weight_enc.get_plain_text()
            linear1_mean_bias = linear1_mean_bias_enc.get_plain_text()
            linear2_mean_weight = linear2_mean_weight_enc.get_plain_text()
            linear2_mean_bias = linear2_mean_bias_enc.get_plain_text()
            decision_mean_weight = decision_mean_weight_enc.get_plain_text()
            decision_mean_bias = decision_mean_bias_enc.get_plain_text()

        return(linear1_mean_weight, linear1_mean_bias, linear2_mean_weight, linear2_mean_bias, decision_mean_weight, decision_mean_bias)



def weights_and_bias_of_model(model):
    """
    Returns weights and bias of the input model
    Careful : the model have to follow the specified architecture specified above
    """
    weight_linear1, bias_linear1 = model.linear1.weight.data.clone(), model.linear1.bias.data.clone()
    weight_linear2, bias_linear2 = model.linear2.weight.data.clone(), model.linear2.bias.data.clone()
    weight_decision, bias_decision = model.decision.weight.data.clone(), model.decision.bias.data.clone()
    return(weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision)


def replace_weights_and_bias_of_model(model, new_weight_linear1, new_bias_linear1, new_weight_linear2, new_bias_linear2, new_weight_decision, new_bias_decision):
    """
    Returns model with updated weights and bias from the input weights and bias
    Careful : the model have to follow the specified architecture specified above
    """
    with torch.no_grad():
        model.linear1.weight.data = new_weight_linear1.data.clone()
        model.linear2.weight.data = new_weight_linear2.data.clone()
        model.decision.weight.data = new_weight_decision.data.clone()
        model.linear1.bias.data = new_bias_linear1.data.clone()
        model.linear2.bias.data = new_bias_linear2.data.clone()
        model.decision.bias.data = new_bias_decision.data.clone()
    return(model)


def federative_average(model_dict):
    """
    Computes average of weights updated from all agents
    """
    nb_agents = len(model_dict)
    weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision=weights_and_bias_of_model(model_dict["model0"])

    linear1_mean_weight = torch.zeros(size=weight_linear1.shape)
    linear1_mean_bias = torch.zeros(size=bias_linear1.shape)
    linear2_mean_weight = torch.zeros(size=weight_linear2.shape)
    linear2_mean_bias = torch.zeros(size=bias_linear2.shape)
    decision_mean_weight = torch.zeros(size=weight_decision.shape)
    decision_mean_bias = torch.zeros(size=bias_decision.shape)

    with torch.no_grad():
        for i in range(nb_agents):
            weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision=weights_and_bias_of_model(model_dict["model"+str(i)])
            linear1_mean_weight += weight_linear1
            linear1_mean_bias += bias_linear1
            linear2_mean_weight += weight_linear2
            linear2_mean_bias += bias_linear2
            decision_mean_weight += weight_decision
            decision_mean_bias += bias_decision

        linear1_mean_weight = linear1_mean_weight/nb_agents
        linear1_mean_bias = linear1_mean_bias/nb_agents
        linear2_mean_weight = linear2_mean_weight/nb_agents
        linear2_mean_bias = linear2_mean_bias/nb_agents
        decision_mean_weight = decision_mean_weight/nb_agents
        decision_mean_bias = decision_mean_bias/nb_agents

    return(linear1_mean_weight, linear1_mean_bias, linear2_mean_weight, linear2_mean_bias, decision_mean_weight, decision_mean_bias)

  
  
def mean_average(list_perf, age, intervalle=cfg.MEAN_INTERVAL):
    """TODO ajouter description ou clarification variables"""
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
    """
    Returns:
        - Means of mouses lifetimes at each age
        - Means of mouses lifetimes variances at each age
    """
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
main_model=qlearn.NeuralModel(number_actions=8, input_size=8)
worlds = []
model_dict = dict()
for world_id in range(nb_worlds):
    world = setup.World(filename='resources/world.txt')     #TODO : peut etre créer plusieurs fichiers tkt worlds
    world.add_agents()

    if cfg.LEARNING_MODE in ['Federated_Deep_QLearning', 'Federated_SMPC_Deep_QLearning']:
        model_name = "model"+str(world_id)
        model_info = world.mouse.ai.model
        model_dict.update({model_name : model_info })

    if world_id == 0:
        world.display.activate()
        world.display.speed = cfg.speed
    worlds.append(world)


#Run code

#Start recording training time
start = time.time()

age_max = cfg.MAX_AGE
for age in range(age_max):
    for world_id in range(nb_worlds):
    #for world in worlds:
        worlds[world_id].update(worlds[world_id].mouse.mouseWin, worlds[world_id].mouse.catWin)

        if cfg.LEARNING_MODE in ['Federated_Deep_QLearning', 'Federated_SMPC_Deep_QLearning']:
            model_dict["model"+str(world_id)]=worlds[world_id].mouse.ai.model

    if age % cfg.nb_maj_required_to_update_main_model == 0 and (cfg.LEARNING_MODE in ['Federated_Deep_QLearning', 'Federated_SMPC_Deep_QLearning']):

        #Federative average
        if cfg.LEARNING_MODE == 'Federated_Deep_QLearning':
            linear1_mean_weight, linear1_mean_bias, linear2_mean_weight, linear2_mean_bias, decision_mean_weight, decision_mean_bias = federative_average(model_dict)
        else: #cfg.LEARNING_MODE == 'Federated_SMPC_Deep_QLearning':
            linear1_mean_weight, linear1_mean_bias, linear2_mean_weight, linear2_mean_bias, decision_mean_weight, decision_mean_bias = encrypted_federative_average(model_dict)

        #Update main model according to aggregated agents models weights
        main_model = replace_weights_and_bias_of_model(main_model, linear1_mean_weight, linear1_mean_bias, linear2_mean_weight, linear2_mean_bias, decision_mean_weight, decision_mean_bias)

        #Update of all agents models
        weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision = weights_and_bias_of_model(main_model)
        for world_id2 in range(nb_worlds):
            model_dict["model"+str(world_id2)] = replace_weights_and_bias_of_model(model_dict["model"+str(world_id2)], weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision)
            worlds[world_id2].mouse.ai.model=model_dict["model"+str(world_id2)]

#End time recording
end = time.time()
time_elapsed = end - start


#Get performances through ages
performances = []
for world in worlds:
    performance = world.get_mouse_performance()
    performances.append(performance)

mean_performance, var_performances = compute_mean_var_performance_over_mouses(mouses_performances=performances)
avg_mouse_lifetime = int(np.mean([np.mean(perfs) for perfs in performances]))


#Saving results
#/results/
#np.save(f'../results/{cfg.LEARNING_MODE}-lifetime-{cfg.MAX_AGE}iterations', mean_performance)
#np.save(f'../results/{cfg.LEARNING_MODE}-var-{cfg.MAX_AGE}iterations', var_performances)
#np.save(f'../results/{cfg.LEARNING_MODE}',mean_performance)
path = os.path.join('results','{}-{}_iterations.npz'.format(cfg.LEARNING_MODE,cfg.MAX_AGE)) #TODO : ajouter nb worlds
np.savez(path, lifetime=mean_performance, var=var_performances, time=np.array([time_elapsed]))


#Plot
import matplotlib.pyplot as plt

plt.figure(1)
plt.title(f"Mouses lifetime (moy interval = {cfg.MEAN_INTERVAL})")
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
