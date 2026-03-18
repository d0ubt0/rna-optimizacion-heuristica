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



def crear_heatmap(f, cantidad_puntos: int = 200, min_x: float = -5, max_x: float = 5, min_y: float = -5, max_y: float = 5):
    x = np.linspace(min_x, max_x, cantidad_puntos)
    y = np.linspace(min_y, max_y, cantidad_puntos)
    X, Y = np.meshgrid(x, y)

    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))


    plt.figure()
    cont = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(cont)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Mapa de colores de la función")
    return cont

def crear_animacion(heatmap, trayectoria, nombre=''):
    import matplotlib.animation as animation

    fig = heatmap.figure
    ax = fig.gca()

    trayectoria = np.array(trayectoria)
    puntos = ax.scatter([], [], c='r', s=5)

    def init():
        puntos.set_offsets(np.empty((0, 2)))
        return puntos,

    def update(frame):
        puntos.set_offsets(trayectoria[frame:frame+1])
        return puntos,

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(trayectoria),
        init_func=init,
        blit=True,
        interval=33
    )

    ani.save(f'animacion_{nombre}.gif', writer='pillow')
    #plt.show()



