import numpy as np
from funciones import (
    crear_animacion,
    crear_animacion_3d,
    crear_heatmap,
    heatmap_3d,
    rastrigin,
    rastrigin_gradiente,
    three_hump_camel,
    three_hump_camel_gradiente,
)


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
parametros_iniciales = np.random.random(2) * rastrigin_max * 2 - rastrigin_max

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

# Three hump camel 2D
camel_max = 2
parametros_iniciales = np.random.random(2) * camel_max * 2 - camel_max

iteraciones = 1000

parametros, trayectoria = descenso_por_gradiente(
    three_hump_camel_gradiente,
    parametros_iniciales,
    iteraciones,
    tasa_de_aprendizaje
)

ax_three = crear_heatmap(
    three_hump_camel,
    cantidad_puntos=2000,
    min_x=-camel_max,
    max_x=camel_max,
    min_y=-camel_max,
    max_y=camel_max
)

crear_animacion(ax_three, three_hump_camel, trayectoria, "three_hump_camel")

# Rastrigin 3D (f: R³ → R)
parametros_iniciales_3d = np.random.random(3) * rastrigin_max * 2 - rastrigin_max

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
