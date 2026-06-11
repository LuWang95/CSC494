import pickle

with open("wrong_physics_trainable.pkl", "rb") as f:

    results = pickle.load(f)
    print(results.keys())