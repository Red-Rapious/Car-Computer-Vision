# Car-Computer-Vision
## Détection d'objets dans un flux vidéo pour caméra embarquée sur un véhicule
Mon projet de TIPE pour l'année 2023, visant la création d'un algorithme de détection d'objets dans une image.

### Objectifs
Dans un premier temps, l'objectif est de développer un système de détection de panneaux et autres avec OpenCV, pour faire fonctionner le véhicule autonome créé par le reste de mon groupe.
Dans un second temps, j'essayerai de développer et d'implémenter mon propre algorithme de Viola-Jones et éventuellement de le faire fonctionner sur le robot si les performances le permettent.

### Contenu
- Implémentation simple avec OpenCV de détection de panneaux, voitures, visages et feux : `source/opencv_impl`
- Utilisation de YOLO pour détecter spécifiquement des voitues : `source/yolo_impl`
- Implémentation de zéro de l'algorithme de Viola-Jones pour de la reconnaissance faciale : `source/viola_jones_impl`

## License
Ceci est un projet original d'Antoine Groudiev, sous licence [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).