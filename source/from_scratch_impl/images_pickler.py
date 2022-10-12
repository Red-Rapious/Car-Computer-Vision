import pickle
import glob

def save(data, filename:str) -> None:
        """ Utilise le module Pickle pour sauvegarder le modèle entraîné"""
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    data = []
    pos_folder = glob.glob("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/fullsize_test_images/positives/*")
    neg_folder = glob.glob("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/fullsize_test_images/negatives/*")

        
    for image_path in pos_folder+neg_folder:
        data.append((image_path, 1 if image_path in pos_folder else 0))
        
    save(data, "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/fullsize_test_images/pickle_files/fullsize_test")