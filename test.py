import numpy as np
import pickle

episode_path = "/home/agnel/Projects/rlds_dataset_builder/ur3_vla/data/train/episode1.txt"

data = np.loadtxt(episode_path, delimiter="|")

for i, step in enumerate(data):
    # compute Kona language embedding
    print(step['language_instruction'])