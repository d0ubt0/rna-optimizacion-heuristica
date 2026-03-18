import numpy as np
from funciones import (
    crear_animacion,
    crear_heatmap,
    rastrigin,
    rastrigin_gradiente,
    three_hump_camel,
    three_hump_camel_gradiente,
)


def descenso_por_gradiente(gradiente_funcion, parametros_iniciales, iteraciones, tasa_de_aprendizaje):
    parametros = parametros_iniciales.copy()
    trayectoria = [parametros.copy()]

    for _ in range(iteraciones):
        gradientes = gradiente_funcion(parametros)
        parametros = parametros - tasa_de_aprendizaje * gradientes
        trayectoria.append(parametros.copy())

    return parametros, trayectoria

iteraciones = 100
tasa_de_aprendizaje = 0.001

# Rastrigin
rastrigin_max = 5.12
parametros_iniciales = np.random.random(2) * rastrigin_max * 2 - rastrigin_max

parametros, trayectoria = descenso_por_gradiente(
    rastrigin_gradiente,
    parametros_iniciales,
    iteraciones,
    tasa_de_aprendizaje
)

heatmap_rastrigin = crear_heatmap(
    rastrigin,
    cantidad_puntos=2000,
    min_x=-rastrigin_max,
    max_x=rastrigin_max,
    min_y=-rastrigin_max,
    max_y=rastrigin_max
)

crear_animacion(heatmap_rastrigin, trayectoria, "rastrigin")


# Three hump camel
camel_max = 2
parametros_iniciales = np.random.random(2) * camel_max * 2 - camel_max

iteraciones = 1000

parametros, trayectoria = descenso_por_gradiente(
    three_hump_camel_gradiente,
    parametros_iniciales,
    iteraciones,
    tasa_de_aprendizaje
)

heatmap_three = crear_heatmap(
    three_hump_camel,
    cantidad_puntos=2000,
    min_x=-camel_max,
    max_x=camel_max,
    min_y=-camel_max,
    max_y=camel_max
)

crear_animacion(heatmap_three, trayectoria, "three_hump_camel")
