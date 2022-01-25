# coding:utf-8

import random
import config as cfg
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralModel(nn.Module):
    def __init__(self, input_size, number_actions):
        super().__init__()
        self.dropout = nn.Dropout(.2)
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.decision = nn.Linear(256, number_actions)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        out = self.decision(x)
        #On ne met pas de couche softmax ici car cela est censé être déja inclu dans 'criterion' cross entropy lors de la phase d'entrainement
        return out          #self.decision(drop.view(x.size(0), -1))


class QLearn:
    """
    Q-learning:
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s', a') - Q(s,a))

        * alpha is the learning rate.
        * gamma is the value of the future reward.
    It use the best next choice of utility in later state to update the former state.
    """
    def __init__(self, actions, input_size, alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions  # collection of choices
        self.epsilon = epsilon  # exploration constant
        self.state_size = input_size

        self.device = torch.device('cpu') # can be changed to cuda if available
        # create main model
        self.model = NeuralModel(input_size=input_size, number_actions=len(actions))
        self.model.to(self.device)

    def _get_q(self, state):
        """Returns : actions values distribution according to one input state"""
        with torch.no_grad():
            out = self.model(state)
            probas = nn.Softmax(dim=-1)(out)
        return probas   #softmax output

    def _convertToTorchModelInput(self,input):
        """Returns a torch tensor of batch size 1 from a single sample input (array)"""
        #TODO : careful with dtype torch.float instead of torch.long
        #torch.LongTensor assigned to device cpu by default (on the other hand torch.cuda.LongTensor assigned to device gpu by default)
        return torch.tensor(np.array([input]), dtype=torch.float, device=self.device)   #Faster if we convert in array than just a list

    # When in certain state, find the best action while explore new grid by chance.
    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            """
            Standard - DQN
            DQN chooses the max Q value among next actions
            selection and evaluation of action is on the target Q Network
            Q_max = max_a' Q_target(s', a')
            """
            state = self._convertToTorchModelInput(state)
            q = self._get_q(state)
            q = q[0]    #get the q of the first (and only one) sample

            action = torch.argmax(q)
        return action

    # learn
    def learn(self, state, action, next_state, reward, is_last_state):
        """Stochastic Learning"""

        # Model predictions (softmax arrays outputs)
        state = self._convertToTorchModelInput(state)
        next_state = self._convertToTorchModelInput(next_state)
        target = self._get_q(state)
        target_next = self._get_q(next_state)

        # Correction on the target Q value for the action used
        if is_last_state:
            target[0][action] += self.alpha*reward #= reward
        else:
            target[0][action] += self.alpha*(reward + self.gamma * (torch.max(target_next[0]))-target[0][action])  #= reward + self.gamma * (torch.max(target_next[0]))
            #Reward propagation through states
        target = nn.Softmax(dim=-1)(target)
        # Train the Neural Network with batches
        self.fit(model=self.model, x=state, y=target)

    def fit(self, model, x, y, epochs=1):
        """
        Stochastic gradient descent : 1 epoch with batch size 1 (1 sample only)

        TODO : stocker les exemples pour un apprentissage par batch plus long ?
        """

        criterion = nn.CrossEntropyLoss()           # fonction de loss à appliquer
        optimizer = optim.Adam(model.parameters())

        for epoch in range(epochs):                 # boucle des époques
            total_loss = 0                          # loss (moyenne) de l'époque
            num = 0

            #for x, y in train_loader:               # charge un batch
            optimizer.zero_grad()                   # par défault, le gradient est cumulatif -> on le remet à 0

            y_scores = model(x)                     # prédictions (forward) avec paramètres actuels
            loss = criterion(y_scores, y)           # différence entre scores prédits et référence (inclut softmax)
            loss.backward()                         # calcule le gradient
            optimizer.step()                        # applique le gradient au modèle

            total_loss += loss.item()               # loss cumulée (pour affichage)
            num += len(y)                           # nombre d'exemples traités
            print(epoch, total_loss / num)
            #if epoch % (epochs // 10) == 0:         # affiche loss toutes les (époques/10) époques
            #print(epoch, total_loss / num)