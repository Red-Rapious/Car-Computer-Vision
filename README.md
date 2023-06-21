# Car-Computer-Vision
## Détection d'objets dans un flux vidéo pour caméra embarquée sur un véhicule
Mon projet de TIPE pour l'année 2023, visant la création d'un algorithme de détection d'objets dans une image. Le rapport complet est disponible [ici](demonstrations/rapport-ens/rapport-ens.pdf).

### Contenu
- Implémentation simple avec OpenCV de détection de panneaux, voitures, et visages, utilisant l'algorithme de **Viola-Jones** : `source/opencv_impl/viola_jones`
- Implémentation avec OpenCV de la même détection avec **YOLO** : `source/opencv_impl/yolo`
- Implémentation de zéro de l'algorithme de Viola-Jones pour de la reconnaissance faciale : `source/from_scratch_impl`
- Divers scripts de comparaison des performances de processeurs : `source/stress_test`
- Scipts pour transformer des images diverses en images formatées pour l'apprentissage : `source/training_data_processing`

## License
Ceci est un projet original d'Antoine Groudiev, sous licence [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).