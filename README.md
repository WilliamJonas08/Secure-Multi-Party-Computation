# Secure-MultiParty-Computation

# Présentation projet
===================================================

Ce projet a été réalisé dans le cadre d'un projet de mon master Intelligence Artificielle et Apprentissage Automatique.

Criteo est une entreprise française de reciblage publicitaire sur internet au chiffre d'affaires de 150 milliards d'euros, diffusant plus de 4 milliards d'annonces chaque jour.

L'objectif du projet (orienté recherche) est d’évaluer la pertinence, pour le domaine du Machine Learning, de la méthode cryptographique appelée Secure Multiparty Computation (sMPC) dans le cadre d’une méthode de Fédérative Learning sur un algorithme de bandits.

## Les principales interrogations sont:
- Cette méthode permet elle d’augmenter la précision des modèles ?
- Cette méthode consomme elle moins d’énergie ? (Données non exportées sur le cloud)
- Quelles sont les limites de cette méthode ?

Nous nous appuyons sur la librairie open-source Crypten. Il s'agit d'un framework construit sur PyTorch pour faciliter la recherche en apprentissage automatique sécurisé et préservant la vie privée. Crypten met en œuvre la méthode sMPC, qui crypte l'information en divisant les données entre plusieurs parties, qui peuvent chacune effectuer des calculs sur leur part mais ne sont pas capables de lire les données originales.

## 📌 La stratégie utilisée consiste à confronter les 3 modèles suivants :
- Algorithme de Bandits ‘classique’
- Algorithme de Bandits entrainé via Federated Learning
- Algorithme de Bandits entrainé via Federated Learning avec méthode sMPC


QLearningMouse
===================================================

<b>QLearningMouse</b>  is a small cat-mouse-cheese game based on [Q-Learning](https://en.wikipedia.org/wiki/Q-learning). The original version is by [vmayoral](https://github.com/vmayoral): [basic_reinforcement_learning:tutorial1](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial1), I reconstructed his code to make the game more configurable, and what different most is that I use breadth-first-search([BFS](https://en.wikipedia.org/wiki/Breadth-first_search)) when cat chasing the AI mouse, so the cat looks much more brutal :P 

## About the game
Cat always chase the mouse in the shortest path, however the mouse first does not know the danger of being eaten. 
* <b>Mouse win</b> when eating the cheese and earns rewards value of 50, then a new cheese will be produced in a random grid.
* <b>cat win</b> when eating the mouse, the latter will gain rewards value of -100 when dead. Then it will relive in a random grid.

## Algorithm  
The basic algorithm of Q-Learning is:  
```
Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s', a') - Q(s,a))
#UPDATE TO CONTEXTUAL BANDITS
```
    
```alpha``` is the learning rate.
```gamma``` is the value of the future reward.
It use the best next choice of utility in later state to update the former state. 

Learn more about Q-Learning:  
1. [The Markov Decision Problem : Value Iteration and Policy Iteration](http://ais.informatik.uni-freiburg.de/teaching/ss03/ams/DecisionProblems.pdf)  
2. [ARTIFICIAL INTELLIGENCE FOUNDATIONS OF COMPUTATIONAL AGENTS : 11.3.3 Q-learning](http://artint.info/html/ArtInt_265.html)


## Example
Below we present a *mouse player* after **300 generations** of reinforcement learning:  
* blue is for mouse.
* black is for cat.
* orange is for cheese.

![](resources/snapshot1.gif)

After **339300 generations**:  

![](resources/snapshot2.gif)


## Reproduce it yourself

```bash
git clone https://github.com/fancoo/QLearningMouse
cd QLearningMouse
python greedyMouse.py
```


## Maj fed learning
pour update : obligé de :
- itérer sur les mondes
- itérer sur chacun de leur agents
- maj selon l'update commune à réaliser

Algo de bandits correspond aux algo de TD Learning ? (savoir si on doit adapter le code et générer des contextes)


Separation taches : 
- class Worlds
- algo bandit/TD Learning
- mise en place/ génération contextes
