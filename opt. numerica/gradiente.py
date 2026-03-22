import numpy as np
from funciones import (
    crear_animacion,
    crear_animacion_3d,
    crear_heatmap,
    heatmap_3d,
    random_seed,
    rastrigin,
    rastrigin_gradiente,
    rosenbrock,
    rosenbrock_gradiente,
)

rng = np.random.default_rng(random_seed)

def descenso_por_gradiente(gradiente_funcion, parametros_iniciales, iteraciones, tasa_de_aprendizaje):
    parametros = parametros_iniciales.copy()
    trayectoria = [parametros.copy()]
    print("Start GD")
    for _ in range(iteraciones):
        print(f"step: {_}") if _ % 10 == 0 else 0
        gradientes = gradiente_funcion(parametros)
        parametros = parametros - tasa_de_aprendizaje * gradientes
        trayectoria.append(parametros.copy())
    print("END GD")

    return parametros, trayectoria

iteraciones = 100
tasa_de_aprendizaje = 0.001

# Rastrigin 2D
rastrigin_max = 5.12
parametros_iniciales = rng.random(2) * rastrigin_max * 2 - rastrigin_max

parametros, trayectoria = descenso_por_gradiente(
    rastrigin_gradiente,
    parametros_iniciales,
    iteraciones,
    tasa_de_aprendizaje
)

ax_rastrigin = crear_heatmap(
    rastrigin,
    cantidad_puntos=2000,
    min_x=-rastrigin_max,
    max_x=rastrigin_max,
    min_y=-rastrigin_max,
    max_y=rastrigin_max
)

crear_animacion(ax_rastrigin, rastrigin, trayectoria, "rastrigin")


# Rastrigin 3D (f: R³ → R)
parametros_iniciales_3d = rng.random(3) * rastrigin_max * 2 - rastrigin_max

parametros_3d, trayectoria_3d = descenso_por_gradiente(
    rastrigin_gradiente,
    parametros_iniciales_3d,
    iteraciones,
    tasa_de_aprendizaje
)

ax_rastrigin_3d = heatmap_3d(
    rastrigin,
    cantidad_puntos=30,
    min_x=-rastrigin_max,
    max_x=rastrigin_max,
    min_y=-rastrigin_max,
    max_y=rastrigin_max,
    min_z=-rastrigin_max,
    max_z=rastrigin_max
)
crear_animacion_3d(ax_rastrigin_3d, rastrigin, trayectoria_3d, "rastrigin_3d")

# Rosenbrock 2D
rosenbrock_max = 2
parametros_iniciales_rosen = rng.random(2) * rosenbrock_max * 2 - rosenbrock_max

parametros_rosen, trayectoria_rosen = descenso_por_gradiente(
    rosenbrock_gradiente,
    parametros_iniciales_rosen,
    iteraciones,
    tasa_de_aprendizaje
)

ax_rosen = crear_heatmap(
    rosenbrock,
    cantidad_puntos=2000,
    min_x=-rosenbrock_max,
    max_x=rosenbrock_max,
    min_y=-rosenbrock_max,
    max_y=rosenbrock_max
)

crear_animacion(ax_rosen, rosenbrock, trayectoria_rosen, "rosenbrock")


# Rosenbrock 3D (f: R³ → R)
parametros_iniciales_rosen_3d = rng.random(3) * rosenbrock_max * 2 - rosenbrock_max

parametros_rosen_3d, trayectoria_rosen_3d = descenso_por_gradiente(
    rosenbrock_gradiente,
    parametros_iniciales_rosen_3d,
    iteraciones,
    tasa_de_aprendizaje
)

ax_rosen_3d = heatmap_3d(
    rosenbrock,
    cantidad_puntos=30,
    min_x=-rosenbrock_max,
    max_x=rosenbrock_max,
    min_y=-rosenbrock_max,
    max_y=rosenbrock_max,
    min_z=-rosenbrock_max,
    max_z=rosenbrock_max
)

crear_animacion_3d(ax_rosen_3d, rosenbrock, trayectoria_rosen_3d, "rosenbrock_3d")
