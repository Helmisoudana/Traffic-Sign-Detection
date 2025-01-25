Traffic Sign Detection
Traffic Sign Detection est un projet de détection des panneaux de signalisation à l'aide d'un modèle de deep learning. Ce projet utilise un modèle pré-entraîné basé sur TensorFlow et OpenCV pour détecter et classer différents types de panneaux de signalisation en temps réel.

Table des matières
Aperçu du projet
Fonctionnalités
Prérequis
Installation
Utilisation
Structure du projet
Contributions
License
Aperçu du projet
Ce projet a été conçu pour détecter les panneaux de signalisation dans des images ou des vidéos. Il utilise OpenCV pour le traitement des images et TensorFlow pour appliquer le modèle de détection des panneaux. L'objectif est d'automatiser le processus de reconnaissance des panneaux dans un contexte de conduite autonome ou d'assistance à la conduite.

Le modèle utilisé est un CNN (Convolutional Neural Network) pré-entraîné sur le dataset GTSRB (German Traffic Sign Recognition Benchmark). Ce dataset contient des images de panneaux de signalisation dans divers environnements, ce qui permet d'obtenir une détection fiable dans des conditions variées.

Fonctionnalités
Détection en temps réel des panneaux de signalisation à partir d'une image ou d'une vidéo.
Classification des panneaux détectés en fonction de leur type (par exemple, stop, limitation de vitesse, etc.).
Interface graphique pour afficher les panneaux détectés sur l'image d'entrée.
Possibilité de tester avec des images locales ou des flux vidéo en temps réel à partir de la webcam.
Prérequis
Avant de commencer, assurez-vous que votre environnement répond aux exigences suivantes :

Python 3.x
TensorFlow 2.x
OpenCV
Numpy
Matplotlib (pour visualisation)
Bibliothèques requises
Les bibliothèques nécessaires peuvent être installées avec pip :

bash
Copier
Modifier
pip install tensorflow opencv-python numpy matplotlib
Installation
Clonez ce dépôt sur votre machine locale :

bash
Copier
Modifier
git clone https://github.com/Helmisoudana/Traffic-Sign-Detection.git
Accédez au répertoire du projet :

bash
Copier
Modifier
cd Traffic-Sign-Detection
Installez les dépendances via pip :

bash
Copier
Modifier
pip install -r requirements.txt
Si le fichier requirements.txt n'existe pas encore, vous pouvez installer manuellement les bibliothèques avec les commandes mentionnées plus haut.

Téléchargez le modèle pré-entraîné (si ce n'est pas déjà inclus dans le dépôt) et placez-le dans le répertoire approprié.

Utilisation
Exécution avec une image locale
Pour tester la détection sur une image locale, exécutez le script suivant :

bash
Copier
Modifier
python detect_traffic_sign.py --image <chemin-vers-l-image>
Cela ouvrira une fenêtre avec l'image affichant les panneaux détectés.

Exécution avec la webcam
Pour tester la détection en temps réel avec votre webcam, utilisez :

bash
Copier
Modifier
python detect_traffic_sign.py --webcam
Cela activera votre caméra et affichera les résultats en temps réel.

Arguments
--image <chemin-vers-l-image> : Spécifie le chemin d'une image locale à tester.
--webcam : Active la détection en temps réel avec la webcam.

detect_traffic_sign.py : Le script qui effectue la détection en temps réel ou sur des images locales.
model/traffic_sign_model.h5 : Le modèle de deep learning pré-entraîné.
utils.py : Contient les fonctions pour charger les images, traiter les résultats et afficher les résultats.
Contributions
Si vous souhaitez contribuer à ce projet, vous pouvez suivre ces étapes :

Fork ce dépôt.
Créez une branche pour votre fonctionnalité (git checkout -b feature/ma-fonctionnalité).
Commitez vos modifications (git commit -am 'Ajout de ma fonctionnalité').
Poussez votre branche (git push origin feature/ma-fonctionnalité).
Ouvrez une Pull Request pour discuter des modifications.
License
Ce projet est sous la MIT License. Voir le fichier LICENSE pour plus de détails.

