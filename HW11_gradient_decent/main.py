import numpy as np
import matplotlib.pyplot as plt
# Define the functions
def function1(x, y):
    return x**2 + y**2
def function2(x, y):
    return x**4 + y**4
# Define the partial derivatives of the functions
def gradient1(x, y):
    return np.array([2*x, 2*y])
def gradient2(x, y):
    return np.array([3*x**3, 3*y**3])
# Gradient Descent function
def gradient_descent(func, gradient, initial_point, learning_rate, num_iterations):
    x, y = initial_point
    x_values, y_values = [x], [y]
    for _ in range(num_iterations):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values
# Plotting the function contours
def plot_contour(func, x_values, y_values, ax, title):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    ax.contour(X, Y, Z, levels=np.logspace(-0.5, 3.5, 20), cmap='jet')
    # ax.contour(X, Y, Z, levels=np.logspace(-0.05, 5.5, 20), cmap='jet')

    ax.plot(x_values, y_values, color='red', marker='o')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
# Running Gradient Descent on both functions with different initial points and learning rates
initial_points = [(4, 4), (-3, -3)]
learning_rates = [0.01, 0.03]
num_iterations = 100
#Plotting
fig, axes0 = plt.subplots(1, len(initial_points), figsize=(5, 5))  # One row for subplots
fig.tight_layout(h_pad=1, w_pad=1)
for i, initial_point in enumerate(initial_points):
    learning_rate = learning_rates[i]
    x_values, y_values = gradient_descent(function1, gradient1, initial_point, learning_rate, num_iterations)
    plot_contour(function1, x_values, y_values, axes0[i], f"function1: x^2 +y^2, Initial Point:{initial_point}, Learning Rate:{learning_rate}")
fig, axes1 = plt.subplots(1, len(initial_points), figsize=(5, 5))  # One row for subplots
fig.tight_layout(h_pad=1, w_pad=1)
for i, initial_point in enumerate(initial_points):
    learning_rate = learning_rates[i]
    x_values, y_values = gradient_descent(function2, gradient2, initial_point, learning_rate, num_iterations)
    plot_contour(function2, x_values, y_values, axes1[i], f"function2 :x^4 +y^4, Initial Point:{initial_point}, Learning Rate:{learning_rate}")
plt.show()
