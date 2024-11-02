import ctrnn
import matplotlib.pyplot as plt
import numpy as np

# EXPERIMENT PARAMETERS
size = 10
duration = 100
stepsize = 0.01
time = np.arange(0.0, duration, stepsize)

# DEFINITION OF THE OBSERVABLE
def synchronization(er):
    nn = ctrnn.CTRNN(size)
    nn.randomizeParameters()

    # Set correct proportions of excitatory/inhibitory connections
    for i in range(size):
        for j in range(size):
            if np.random.random() < er:
                nn.Weights[i, j] = np.random.random() * 10.0 - 5.0
            else:
                nn.Weights[i, j] = 0.0

    nn.initializeState(np.zeros(size))
    outputs = np.zeros((len(time), size))

    # Run transient
    for t in time:
        nn.step(stepsize)

    # Run simulation and record outputs
    step = 0
    for t in time:
        nn.step(stepsize)
        outputs[step] = nn.Outputs
        step += 1

    if np.any(np.std(outputs, axis=0) == 0):
        mean_correlation = np.nan  # or set to zero or another indicator if needed
    else:
        correlation_matrix = np.corrcoef(outputs.T)
        upper_triangle_indices = np.triu_indices(size, k=1)
        mean_correlation = np.mean(correlation_matrix[upper_triangle_indices])


    return mean_correlation

# ITERATE THROUGH DIFFERENT PROPORTIONS
reps = 100
steps = 11
errange = np.linspace(0.0, 1.0, steps)

data = np.zeros((steps, reps))
k = 0
for er in errange:
    for r in range(reps):
        data[k][r] = synchronization(er)
    k += 1

# Visualize the results
plt.plot(errange, np.mean(data, axis=1), 'ko')
plt.plot(errange, np.mean(data, axis=1) + np.std(data, axis=1) / np.sqrt(reps), 'k.')
plt.plot(errange, np.mean(data, axis=1) - np.std(data, axis=1) / np.sqrt(reps), 'k.')
plt.xlabel("Proportion of excitatory connections")
plt.ylabel("Mean synchronization (pairwise correlation)")
plt.show()

# # Compute pairwise correlation between each pair of neuron outputs
# outputs += np.random.normal(0, 1e-6, outputs.shape)  # Add small noise to avoid zero variance
# variances = np.var(outputs, axis=0)

# # Remove neurons with zero variance (constant output neurons)
# valid_indices = np.where(variances > 1e-10)[0]
# filtered_outputs = outputs[:, valid_indices]

# if filtered_outputs.shape[1] > 1:
#     correlation_matrix = np.corrcoef(filtered_outputs.T)
#     upper_triangle_indices = np.triu_indices(len(valid_indices), k=1)
#     mean_correlation = np.mean(correlation_matrix[upper_triangle_indices])
# else:
#     mean_correlation = 0  # Default if insufficient neurons are left

# return mean_correlation
# Compute pairwise correlation only if there’s variability in outputs
