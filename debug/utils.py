import matplotlib.pyplot as plt
import math

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