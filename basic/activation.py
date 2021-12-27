import numpy as np
from matplotlib import pyplot as plt

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# tanh
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x)+np.exp(-x))

# ReLU(Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU
def eaky_relu(x):
    return np.maximum(0.01 * x, x)

# ELU
def elu(x, alp):
    return (x > 0) * x + (x <= 0) + (alp * (np.exp(x) - 1))

x = np.arange(float(-10), float(10), float(0.01))

# draw sigmoid
y = sigmoid(x)
plt.title("sigmoid")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.plot(x, y)
plt.show()

# draw tanh
y = tanh(x)
plt.title("tanh")
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.plot(x, y)
plt.show()

# draw ReLU(Rectified Linear Unit)
y = tanh(x)
plt.title("ReLU")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.plot(x, y)
plt.show()

# draw Leaky ReLU
y = tanh(x)
plt.title("Leaky ReLU")
plt.xlabel("x")
plt.ylabel("Leaky ReLU(x)")
plt.plot(x, y)
plt.show()






