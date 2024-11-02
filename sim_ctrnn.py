import ctrnn
import matplotlib.pyplot as plt
import numpy as np

# Parameters
size = 10 
duration = 100
stepsize = 0.01
activation_functions = [ctrnn.sigmoid, ctrnn.tanh, ctrnn.relu, ctrnn.sine]
activation_names = ["Sigmoid", "Tanh", "ReLU", "Sine"]

# Data
for i, act in enumerate(activation_functions): 
    time = np.arange(0.0,duration,stepsize)
    outputs = np.zeros((len(time),size))
    states = np.zeros((len(time),size))

    # Initialization
    nn = ctrnn.CTRNN(size, act)

    # Neural parameters at random
    nn.randomizeParameters()

    # Initialization at zeros or random
    nn.initializeState(np.zeros(size))
    #nn.initializeState(np.random.random(size=size)*20-10)

    # Run simulation
    step = 0
    for t in time:
        nn.step(stepsize)
        states[step] = nn.States
        outputs[step] = nn.Outputs
        step += 1

    # How much is the neural activity changing over time
    activity = np.sum(np.abs(np.diff(outputs,axis=0)))/(duration*size*stepsize)
    print("Overall activity: ",activity)

    # Plot activity
    plt.plot(time,outputs)
    plt.xlabel("Time")
    plt.ylabel("Outputs")
    plt.title(f"Neural output activity: {activation_names[i]}")
    plt.show()

    # Plot activity
    plt.plot(time,states)
    plt.xlabel("Time")
    plt.ylabel("States")
    plt.title(f"Neural state activity: {activation_names[i]}")
    plt.show()

# # Save CTRNN parameters for later
# nn.save("ctrnn")
