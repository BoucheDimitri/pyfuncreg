import pickle
import matplotlib.pyplot as plt

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-03-2020_11-02/outputs/output_noise/"
with open(path + "full.pkl", "rb") as inp:
    noise_levels, scores = pickle.load(inp)

plt.figure()
for key in scores.keys():
    plt.plot(noise_levels, scores[key], label="N=" + str(key))
plt.legend()