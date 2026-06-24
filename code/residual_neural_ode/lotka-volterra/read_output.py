import pickle
from code.error_propagation.visualize_lotka_volterra import visualize_results

with open("with_carrying_fixed.pkl", "rb") as f:

    results = pickle.load(f)
    visualize_results(results)