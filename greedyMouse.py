# coding:utf-8

import random
import setup
import qlearn
import config as cfg
import torch

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

def weights_and_bias_of_model(model):
    weight_linear1, bias_linear1=model.linear1.weight.data.clone(), model.linear1.bias.data.clone()
    weight_linear2, bias_linear2=model.linear2.weight.data.clone(), model.linear2.bias.data.clone()
    weight_decision, bias_decision=model.decision.weight.data.clone(), model.decision.bias.data.clone()
    return(weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision)

def replace_weights_and_bias_of_model(model, new_weight_linear1, new_bias_linear1, new_weight_linear2, new_bias_linear2, new_weight_decision, new_bias_decision):
    with torch.no_grad():
        model.linear1.weight.data = new_weight_linear1.data.clone()
        model.linear2.weight.data = new_weight_linear2.data.clone()
        model.decision.weight.data = new_weight_decision.data.clone()
        model.linear1.bias.data = new_bias_linear1.data.clone()
        model.linear2.bias.data = new_bias_linear2.data.clone()
        model.decision.bias.data = new_bias_decision.data.clone()
    return(model)

def federative_average(model_dict):
    nb_agents=len(model_dict)
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

#Initialisation worlds
nb_worlds = cfg.number_of_worlds
main_model=qlearn.NeuralModel(number_actions=8, input_size=8)
worlds = []
model_dict = dict()
for world_id in range(nb_worlds):
    world = setup.World(filename='resources/world.txt')     #TODO : peut etre créer plusieurs fichiers tkt worlds
    world.add_agents()

    model_name="model"+str(world_id)
    model_info=world.mouse.ai.model
    model_dict.update({model_name : model_info })

    if world_id ==0:
        world.display.activate()
        world.display.speed = cfg.speed
    worlds.append(world)

#Run code
#while 1:
age_max = cfg.MAX_AGE
for i in range(age_max):
    for j in range(nb_worlds):
    #for world in worlds:
        worlds[j].update(worlds[j].mouse.mouseWin, worlds[j].mouse.catWin)
        model_dict["model"+str(j)]=worlds[j].mouse.ai.model
    if i%cfg.update_of_main_model==0:
        #Federative average
        linear1_mean_weight, linear1_mean_bias, linear2_mean_weight, linear2_mean_bias, decision_mean_weight, decision_mean_bias=federative_average(model_dict)
        main_model=replace_weights_and_bias_of_model(main_model, linear1_mean_weight, linear1_mean_bias, linear2_mean_weight, linear2_mean_bias, decision_mean_weight, decision_mean_bias)
        #MàJ des modèles de tous les agents
        weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision=weights_and_bias_of_model(main_model)
        for j in range(nb_worlds):
            model_dict["model"+str(j)]=replace_weights_and_bias_of_model(model_dict["model"+str(j)], weight_linear1, bias_linear1, weight_linear2, bias_linear2, weight_decision, bias_decision)
            worlds[j].mouse.ai.model=model_dict["model"+str(j)]
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
