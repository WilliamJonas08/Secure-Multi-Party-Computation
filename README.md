# Secure-MultiParty-Computation

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
