# coding:utf-8

import random
import config as cfg
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralModel(nn.Module):
    def __init__(self, input_size, number_actions):
        super().__init__()
        self.hidden_size = number_actions*4
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        #self.linear2 = nn.Linear(128, 256)
        self.decision = nn.Linear(self.hidden_size, number_actions)
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        #x = self.linear2(x)
        #x = F.relu(x)
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
        self.state_size = input_size
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99  # exploration constant if epsilon_decay = 1

        self.memory = deque(maxlen=1000)
        self.batch_size = 64
        self.train_start = 70

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
        return torch.tensor(np.array(input), dtype=torch.float, device=self.device)   #Faster if we convert in array than just a list


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
            state = self._convertToTorchModelInput([state])
            q = self._get_q(state)
            q = q[0]    #get the q of the first (and only one) sample

            action = torch.argmax(q)
        return action

    # learn
    def learn(self, state, action, next_state, reward, is_last_state):
        """
        Put current experience in memory
        Train the model on randomy sampled experiences from memory (number = batchsize)
        """

        #Remember experience
        experience = state, action, next_state, reward, is_last_state
        self._remember(experience)

        #Train model
        if len(self.memory) >= self.train_start:
            self._trainModel()


    def _trainModel(self):
        """Train the model on randomy sampled experiences from memory (number = batchsize)"""
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, self.batch_size)     #min(len(self.memory), self.batch_size)
        minibatch = np.array(minibatch)
        #print('shape minibatch :', minibatch.shape, type(minibatch), minibatch.dtype)
        #When len(memory) < batch size: we train on the full memory

        #TODO : use index selection instead of loop
        #Reminder : expérience = state, action, reward, next_state, done
        #states = minibatch[:,0].astype(dtype=float)
        #actions = np.array(minibatch[:,1], dtype=np.int)
        #next_states = np.array(minibatch[:,2], dtype=np.int)
        #rewards = np.array(minibatch[:,3], dtype=np.float)
        #is_last_state = np.array(minibatch[:,4])

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, is_last_state = [], [], []
        for i in range(self.batch_size):
            state, action, next_state, reward, done = minibatch[i]
            states[i] = state
            actions.append(action)
            rewards.append(reward)
            next_states[i] = next_state
            is_last_state.append(done)

        # Model predictions (softmax arrays outputs)
        print('shape states 1 :', states.shape, type(states), states.dtype)
        states = self._convertToTorchModelInput(states)
        print('shape states 2 :', states.shape)
        next_states = self._convertToTorchModelInput(next_states)
        targets = self._get_q(states)
        targets_next = self._get_q(next_states)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if is_last_state[i]:
                #print("REWARD :", rewards[i], type(rewards[i]), rewards[i].dtype)
                targets[i][actions[i]] = self.alpha * rewards[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                #print("REWARD :", rewards[i], type(rewards[i]), rewards[i].dtype)
                targets[i][actions[i]] += self.alpha * (rewards[i] + self.gamma * (torch.max(targets_next[i])) - targets[i][actions[i]])

            #targets[i][actions[i]]  = nn.Softmax(dim=-1)(targets[i][actions[i]])
        # Train the Neural Network with batches
        self._fit(model=self.model, X=states, Y=targets)

        # Model predictions (softmax arrays outputs)
        #state = self._convertToTorchModelInput(state)
        #next_state = self._convertToTorchModelInput(next_state)
        #target = self._get_q(state)
        #target_next = self._get_q(next_state)

        # Correction on the target Q value for the action used
        #if is_last_state:
        #    target[0][action] = reward
        #else:
        #    target[0][action] = reward + self.gamma * (torch.max(target_next[0]))   #Reward propagation through states

        # Train the Neural Network with batches
        #self.fit(model=self.model, x=state, y=target)


    def _remember(self, experience):
        """Put current experience in memory"""
        self.memory.append(experience)
        if len(self.memory) > self.train_start: #Si nombre de couples état action à mémoriser atteint (=train_start) -> maj Epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay  #Update epsilon


    def _fit(self, model, X, Y, epochs=1):
        criterion = nn.CrossEntropyLoss()           # fonction de loss à appliquer
        optimizer = optim.Adam(model.parameters())

        for epoch in range(epochs):                 # boucle des époques
            total_loss = 0                          # loss (moyenne) de l'époque
            num = 0

            #for x, y in zip(X, Y):                # charge un batch
            optimizer.zero_grad()             # par défault, le gradient est cumulatif -> on le remet à 0

            Y_scores = model(X)               # prédictions (forward) avec paramètres actuels
            loss = criterion(Y_scores, Y)     # différence entre scores prédits et référence (inclut softmax)
            loss.backward()                   # calcule le gradient
            optimizer.step()                  # applique le gradient au modèle

            total_loss += loss.item()         # loss cumulée (pour affichage)
            num += len(Y)                     # nombre d'exemples traités
            print(epoch, total_loss / num)
            #if epoch % (epochs // 10) == 0:       # affiche loss toutes les (époques/10) époques
                #print(epoch, total_loss / num)