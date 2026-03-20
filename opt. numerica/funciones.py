import matplotlib.pyplot as plt
import numpy as np


def rastrigin(parametros: np.ndarray, A=10) -> float:
    parametros_copy = parametros.copy()
    d = parametros_copy.size
    resultado = A * d + np.sum(
        parametros_copy**2 - A * np.cos(2 * np.pi * parametros_copy)
    )

    return resultado


def three_hump_camel(parametros: np.ndarray) -> float:
    parametros_copy = parametros.copy()
    d = parametros_copy.size
    if d != 2:
        raise ValueError("Tiene que ser der dos dimensiones")
    x1 = parametros_copy[0]
    x2 = parametros_copy[1]

    resultado = 2 * (x1**2) - 1.05 * (x1**4) + ((x1**6) / 6) + (x1 * x2) + (x2**2)

    return resultado


def rastrigin_gradiente(parametros: np.ndarray, A=10) -> np.ndarray:
    parametros_copy = parametros.copy()
    gradient = 2 * parametros_copy + 2 * A * np.pi * np.sin(2 * np.pi * parametros_copy)
    return gradient


def three_hump_camel_gradiente(parametros: np.ndarray) -> np.ndarray:
    parametros_copy = parametros.copy()
    if parametros_copy.size != 2:
        raise ValueError("Tiene que ser de dos dimensiones")
    x1, x2 = parametros_copy[0], parametros_copy[1]

    df_dx1 = 4 * x1 - 4.2 * x1**3 + x1**5 + x2
    df_dx2 = x1 + 2 * x2

    return np.array([df_dx1, df_dx2])



def crear_heatmap(f, cantidad_puntos: int = 200,
                  min_x: float = -5, max_x: float = 5,
                  min_y: float = -5, max_y: float = 5):
    """
    Visualizes a function f: R² → R as a filled-contour heatmap.

    Samples a 2D grid and plots the function values as colored contours.

    Parameters:
        f               : function accepting np.array([x, y]) → scalar
        cantidad_puntos : grid resolution per axis (total points = n²)
        min_x … max_y   : axis ranges

    Returns:
        ax : the 2D Axes object (use ax.figure to access the figure)
    """
    x = np.linspace(min_x, max_x, cantidad_puntos)
    y = np.linspace(min_y, max_y, cantidad_puntos)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    cont = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(cont, shrink=0.8, aspect=15, pad=0.05, label='f(x, y)')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mapa de colores — f: R² → R')

    return ax

def crear_animacion(ax, f, trayectoria, nombre=''):
    """
    Animates a gradient-descent trajectory in 2D input space for f: R² → R.

    The two spatial axes represent the two input variables (x, y).
    The current point and its trail are drawn over the heatmap produced by
    crear_heatmap.

    Parameters:
        ax          : the 2D Axes returned by crear_heatmap
        f           : the objective function f: R² → R
        trayectoria : list/array of 2D points (N × 2) visited during optimization
        nombre      : name for the output GIF file
    """
    import matplotlib.animation as animation

    fig = ax.figure
    trayectoria = np.array(trayectoria)  # shape: (N, 2)

    # Compute the function value at every trajectory point (for coloring)
    f_values = np.array([f(p) for p in trayectoria])

    # Current-point marker and trail line
    punto, = ax.plot([], [], 'ro', markersize=8, zorder=5)
    linea, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.8, zorder=4)

    def init():
        punto.set_data([], [])
        linea.set_data([], [])
        return punto, linea

    def update(frame):
        # Current point in input space (x, y)
        punto.set_data(
            [trayectoria[frame, 0]],
            [trayectoria[frame, 1]]
        )
        # Trail up to current frame
        linea.set_data(
            trayectoria[:frame + 1, 0],
            trayectoria[:frame + 1, 1]
        )
        return punto, linea

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(trayectoria),
        init_func=init,
        blit=True,
        interval=50
    )

    ani.save(f'animacion_{nombre}.gif', writer='pillow')


def heatmap_3d(f, cantidad_puntos: int = 30,
               min_x: float = -5, max_x: float = 5,
               min_y: float = -5, max_y: float = 5,
               min_z: float = -5, max_z: float = 5,
               umbral_percentil: float = 25):
    """
    Visualizes a function f: R³ → R as a volumetric scatter in 3D input space.

    Samples a 3D grid and plots only the points whose function value falls
    below the given percentile threshold, revealing the basins / low-value
    regions of the function.  Points are colored by f(x, y, z).

    Parameters:
        f                : function accepting np.array([x, y, z]) → scalar
        cantidad_puntos  : grid resolution per axis (total points = n³)
        min_x … max_z    : axis ranges
        umbral_percentil : only points with f-value ≤ this percentile are shown

    Returns:
        ax : the 3D Axes object (use ax.figure to access the figure)
    """
    x = np.linspace(min_x, max_x, cantidad_puntos)
    y = np.linspace(min_y, max_y, cantidad_puntos)
    z = np.linspace(min_z, max_z, cantidad_puntos)

    X, Y, Z = np.meshgrid(x, y, z)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    values = np.array([f(p) for p in points])

    # Keep only the low-value points to reveal the structure
    threshold = np.percentile(values, umbral_percentil)
    mask = values <= threshold

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        points[mask, 0], points[mask, 1], points[mask, 2],
        c=values[mask], cmap='viridis', alpha=0.25, s=12,
        edgecolors='none'
    )
    fig.colorbar(sc, shrink=0.5, aspect=10, pad=0.1, label='f(x, y, z)')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Regiones de bajo valor — f: R³ → R')

    return ax


def crear_animacion_3d(ax, f, trayectoria, nombre=''):
    """
    Animates a gradient-descent trajectory in 3D input space for f: R³ → R.

    The three spatial axes represent the three input variables (x, y, z).
    The current point and its trail are colored by the function value at
    each step, and the camera rotates slowly for a dynamic view.

    Parameters:
        ax          : the 3D Axes returned by heatmap_3d
        f           : the objective function f: R³ → R
        trayectoria : list/array of 3D points (N × 3) visited during optimization
        nombre      : name for the output GIF file
    """
    import matplotlib.animation as animation

    fig = ax.figure
    trayectoria = np.array(trayectoria)  # shape: (N, 3)

    # Compute the function value at every trajectory point (for coloring)
    f_values = np.array([f(p) for p in trayectoria])

    # Current-point marker and trail line
    punto, = ax.plot([], [], [], 'ro', markersize=8, zorder=5)
    linea, = ax.plot([], [], [], 'r-', linewidth=1.5, alpha=0.8, zorder=4)

    def init():
        punto.set_data_3d([], [], [])
        linea.set_data_3d([], [], [])
        return punto, linea

    def update(frame):
        # Current point in input space (x, y, z)
        punto.set_data_3d(
            [trayectoria[frame, 0]],
            [trayectoria[frame, 1]],
            [trayectoria[frame, 2]]
        )
        # Trail up to current frame
        linea.set_data_3d(
            trayectoria[:frame + 1, 0],
            trayectoria[:frame + 1, 1],
            trayectoria[:frame + 1, 2]
        )
        # Slowly rotate the camera
        ax.view_init(elev=30, azim=45 + frame * 270 / len(trayectoria))
        return punto, linea

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(trayectoria),
        init_func=init,
        blit=False,   # 3D axes don't support blitting
        interval=50
    )

    ani.save(f'animacion_3d_{nombre}.gif', writer='pillow')

def rosenbrock(parametros: np.ndarray) -> float:
    parametros_copy = parametros.copy()

    resultado = np.sum(
        100 * (parametros_copy[1:] - parametros_copy[:-1]**2)**2
        + (parametros_copy[:-1] - 1)**2
    )

    return resultado


def rosenbrock_gradiente(parametros: np.ndarray) -> np.ndarray:
    parametros_copy = parametros.copy()
    d = parametros_copy.size

    grad = np.zeros_like(parametros_copy)

    # i = 0
    grad[0] = (
        -400 * parametros_copy[0] * (parametros_copy[1] - parametros_copy[0]**2)
        + 2 * (parametros_copy[0] - 1)
    )

    # 1 <= i <= d-2
    for i in range(1, d - 1):
        grad[i] = (
            -400 * parametros_copy[i] * (parametros_copy[i+1] - parametros_copy[i]**2)
            + 2 * (parametros_copy[i] - 1)
            + 200 * (parametros_copy[i] - parametros_copy[i-1]**2)
        )

    # i = d-1
    grad[-1] = 200 * (parametros_copy[-1] - parametros_copy[-2]**2)

    return grad
