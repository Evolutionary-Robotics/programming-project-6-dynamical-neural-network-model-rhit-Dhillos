# import ctrnn
# import matplotlib.pyplot as plt
# import numpy as np

# # Experiment parameters
# size = 10
# duration = 100
# stepsize = 0.01
# time = np.arange(0.0, duration, stepsize)
# reps = 100

# # Define activation functions and their names
# activation_functions = [ctrnn.sigmoid, ctrnn.tanh, ctrnn.relu, ctrnn.sine]
# activation_names = ["Sigmoid", "Tanh", "ReLU", "Sine"]

# # Run experiment for each activation function
# results = []
# for activation_function, name in zip(activation_functions, activation_names):
#     activity_levels = []
#     for _ in range(reps):
#         nn = ctrnn.CTRNN(size, activation=activation_function)
#         nn.randomizeParameters()
#         nn.initializeState(np.zeros(size))
#         outputs = np.zeros((len(time), size))

#         # Transient phase
#         for t in time:
#             nn.step(stepsize)

#         # Run simulation and capture outputs
#         step = 0
#         for t in time:
#             nn.step(stepsize)
#             outputs[step] = nn.Outputs
#             step += 1

#         # Measure activity
#         activity = np.sum(np.abs(np.diff(outputs, axis=0))) / (duration * size * stepsize)
#         activity_levels.append(activity)
    
#     # Store average activity for each activation function
#     results.append((name, np.mean(activity_levels)))

# # Plot results
# labels, means = zip(*results)
# plt.bar(labels, means, capsize=5)
# print(means)
# plt.xlabel("Activation Function")
# plt.ylabel("Activity in Circuit")
# plt.title("Effect of Activation Function on Proportion of Active Neurons")
# plt.show()




# # Experiment Parameters
# sizes = [1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 500, 1000]  # Different network sizes
# duration = 100
# stepsize = 0.01
# time = np.arange(0.0,duration,stepsize)

# def run_size_experiment(network_size):
#     nn = ctrnn.CTRNN(network_size)
#     nn.randomizeParameters()
#     nn.initializeState(np.zeros(network_size))
#     outputs = np.zeros((len(time), network_size))
 
#     # Run transient
#     for t in time:
#         nn.step(stepsize)

#     # Run simulation
#     step = 0
#     for t in time:
#         nn.step(stepsize)
#         outputs[step] = nn.Outputs
#         step += 1
 
#     activity = np.sum(np.abs(np.diff(outputs, axis=0))) / (duration * network_size * stepsize)
#     return activity

# reps = 100
# # Collect Data
# data = np.zeros((len(sizes), reps))
# for i, network_size in enumerate(sizes):
#  for r in range(reps):
#      data[i][r] = run_size_experiment(network_size)

# # Visualize Results
# plt.plot(sizes, np.mean(data, axis=1), 'ko-')
# plt.xlabel("Number of neurons in the circuit")
# plt.ylabel("Amount of activity in the circuit")
# plt.show()


# import ctrnn
# import matplotlib.pyplot as plt
# import numpy as np

# # Experiment Parameters
# size = 10
# duration = 100
# stepsize = 0.01
# time = np.arange(0.0, duration, stepsize)
# weight_ranges = [(-0.1, 0.1),(-0.5, 0.5),(-0.25, 0.25),(-1, 1),
#                  (-2, 2),(-5, 5),(-10, 10),
#                  (-15, 15),(-20, 20),(-25, 25),(-30, 30),
#                  (-50,50),(-75,75), (-100, 100), (-125, 125),
#                  (-150, 150), (-175, 175), (-200, 200) 
#                 ]
# reps = 100
# # Experiment Function
# def run_weight_range_experiment(weight_range):
#     nn = ctrnn.CTRNN(size)
#     nn.randomizeParameters()
    
#     # Set weights based on the range provided
#     nn.Weights = np.random.uniform(*weight_range, size=(size, size))
#     nn.initializeState(np.zeros(size))
#     outputs = np.zeros((len(time), size))
 
#     # Run transient
#     for t in time:
#         nn.step(stepsize)

#     # Run simulation
#     step = 0
#     for t in time:
#         nn.step(stepsize)
#         outputs[step] = nn.Outputs
#         step += 1
    
#     activity = np.sum(np.abs(np.diff(outputs, axis=0))) / (duration * size * stepsize)
#     return activity

# # Collect Data
# data = np.zeros((len(weight_ranges), reps))
# for i, weight_range in enumerate(weight_ranges):
#  for r in range(reps):
#      data[i][r] = run_weight_range_experiment(weight_range)

# # Visualize Results
# plt.plot([str(wr[1]) for wr in weight_ranges], np.mean(data, axis=1), 'ko-')
# plt.xlabel("Range of weight values")
# plt.ylabel("Amount of activity in the circuit")
# plt.show()

# import ctrnn
# import matplotlib.pyplot as plt
# import numpy as np


# # Experiment parameters
# size = 10
# duration = 100
# stepsize = 0.01
# time = np.arange(0.0, duration, stepsize)
# reps = 100

# # Define the list of activation functions to test
# activation_functions = [ctrnn.sigmoid, ctrnn.tanh, ctrnn.relu, ctrnn.sine]
# activation_names = ["Sigmoid", "Tanh", "ReLU", "Sine"]

# # Run the experiment
# results = []
# for activation_function, name in zip(activation_functions, activation_names):
#     activity_levels = []
#     for _ in range(reps):
#         nn = ctrnn.CTRNN(size, activation=activation_function)
#         nn.randomizeParameters()
#         nn.initializeState(np.zeros(size))

#         # Transient phase to let the network stabilize
#         for t in time:
#             nn.step(stepsize)

#         # Measure the proportion of active neurons over time
#         active_count = 0
#         for t in time:
#             nn.step(stepsize)
#             active_count += np.sum(nn.Outputs > 0.5)  # Threshold for "active"

#         # Calculate the average proportion of active neurons
#         avg_active_proportion = active_count / (len(time) * size)
#         activity_levels.append(avg_active_proportion)
    
#     # Store the results for this activation function
#     results.append((name, np.mean(activity_levels), np.std(activity_levels)))

# # Plot the results
# labels, means, stds = zip(*results)
# plt.bar(labels, means, yerr=stds, capsize=5)
# plt.xlabel("Activation Function")
# plt.ylabel("Proportion of Active Neurons")
# plt.title("Effect of Activation Function on Proportion of Active Neurons")
# plt.show()
