import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import math
import numpy as np

def plot_lines(coord_pairs):
    """
    Plot lines between given points.

    Args:
    coord_pairs (list of tuples): List of coordinate pairs.
                                   Each tuple should contain (x, y) coordinates.
    """
    x_coords, y_coords = zip(*coord_pairs)
    plt.plot(x_coords, y_coords)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Lines between Given Points')
    plt.grid(True)
    plt.show()
    
def draw_vectors(vectors):
    """
    Draw vectors using Matplotlib.

    Args:
    vectors (list of numpy arrays): List containing the vectors as numpy arrays.

    Returns:
    None
    """
    # Extract x and y components from the vectors
    x = [vector[0] for vector in vectors]
    y = [vector[1] for vector in vectors]

    max_norm = math.sqrt(max([xi*xi + yi*yi for xi, yi in vectors]))

    plt.figure()

    # Plot vectors
    plt.quiver([0]*len(vectors), [0]*len(vectors), x, y, angles='xy', scale_units='xy', scale=1)

    # Set limits
    plt.xlim(-max_norm, max_norm)
    plt.ylim(-max_norm, max_norm)

    plt.gca().set_aspect('equal', adjustable='box')

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vector Plot')

    # Show plot
    plt.grid()
    plt.show()

def draw_2D_function(f, bounds=[[-10,10],[-10,10]]):
    x = np.linspace(*bounds[0], 100)
    y = np.linspace(*bounds[1], 100)

    # Create a grid of x and y values
    X, Y = np.meshgrid(x, y)

    Z = np.empty_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = f(X[i, j], Y[i, j])

    min_value = np.min(Z)
    max_value = np.max(Z)

    # Plotting
    plt.figure()
    plt.contourf(X, Y, Z, levels=500, cmap='Greys')  # Contour plot
    cbar = plt.colorbar()  # Add color bar
    cbar.set_ticks([min_value, max_value])  # Set ticks to [1]
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour plot of f(x, y)')
    plt.show()


def draw_1D_function(f, bounds=[-10, 10]):
    x_values = np.linspace(*bounds, 100)
    y_values = []

    for x in x_values:
        y_values.append(f(x))

    # Plotting
    plt.figure()
    plt.plot(x_values, y_values, color='blue')  # Line plot
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('1D plot of f(x)')
    plt.grid(True)
    plt.show()