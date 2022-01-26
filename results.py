import config as cfg
import numpy as np
import matplotlib.pyplot as plt

nb_results_available = 4

max_age = 100
means_lifetime = []
means_var = []
times = []
for learning_mode in cfg.learning_modes[:nb_results_available]:
    mode = learning_mode
    data = np.load(f'results/{learning_mode}-{max_age}_iterations.npz')
    mean_lifetime = data['lifetime']
    mean_var = data['var']
    runtime = data['time'][0]

    means_lifetime.append(mean_lifetime)
    means_var.append(mean_var)
    times.append(runtime)

print("---RUNTIMES---")
for runtime, learning_mode in zip(times,cfg.learning_modes[:nb_results_available]):
    print(f'{learning_mode} runtime : {runtime}s')

plt.figure(1)
plt.title("Mouses lifetime")
for i, learning_mode in enumerate(cfg.learning_modes[:nb_results_available]):
    plt.plot(means_lifetime[i], linewidth=1, label=learning_mode)
plt.legend()

plt.figure(2)
plt.title("Variance of mouses performance")
for i, learning_mode in enumerate(cfg.learning_modes[:nb_results_available]):
    plt.plot(means_var[i], linewidth=1, label=learning_mode)

plt.show()



# import psutil
# # gives a single float value
# print(psutil.cpu_percent())
# # gives an object with many fields
# print(psutil.virtual_memory())
# # you can convert that object to a dictionary
# dict(psutil.virtual_memory()._asdict())
# # you can have the percentage of used RAM
# print(psutil.virtual_memory().percent)
# #79.2
# # you can calculate percentage of available memory
# print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
# #20.8