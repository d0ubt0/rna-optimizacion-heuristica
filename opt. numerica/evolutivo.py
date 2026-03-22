import numpy as np
from funciones import (
    crear_animacion,
    crear_animacion_3d,
    crear_heatmap,
    heatmap_3d,
    random_seed,
    rastrigin,
    rosenbrock,
)

rng = np.random.default_rng(random_seed)

# ─────────────────────────────────────────────
#  ALGORITMO EVOLUTIVO (Estrategia Evolutiva)
# ─────────────────────────────────────────────
def algoritmo_evolutivo(
    funcion_objetivo,
    bounds,
    punto_inicial,          # ← nuevo parámetro
    iteraciones=100,
    tamanio_poblacion=50,
    tasa_mutacion=0.1,
    tasa_cruce=0.7,
    elitismo=2,
):
    dimensiones = len(bounds)
    limites_min = np.array([b[0] for b in bounds])
    limites_max = np.array([b[1] for b in bounds])

    # ── Inicialización: el individuo 0 ES el punto_inicial del GD ────────
    poblacion = limites_min + np.random.random((tamanio_poblacion, dimensiones)) * (
        limites_max - limites_min
    )
    poblacion[0] = np.clip(punto_inicial, limites_min, limites_max)


    trayectoria = []

    print("Start EA")
    for generacion in range(iteraciones):
        fitness = np.array([funcion_objetivo(ind) for ind in poblacion])
        orden = np.argsort(fitness)
        poblacion = poblacion[orden]
        fitness = fitness[orden]

        mejor = poblacion[0].copy()
        trayectoria.append(mejor)

        if generacion % 10 == 0:
            print(f"step: {generacion}  |  mejor fitness: {fitness[0]:.6f}")

        nueva_gen = [poblacion[i].copy() for i in range(elitismo)]

        while len(nueva_gen) < tamanio_poblacion:
            def torneo(k=3):
                competidores = np.random.choice(tamanio_poblacion, k, replace=False)
                ganador = competidores[np.argmin(fitness[competidores])]
                return poblacion[ganador].copy()

            padre1 = torneo()
            padre2 = torneo()

            if np.random.random() < tasa_cruce:
                alpha = np.random.random(dimensiones)
                hijo = alpha * padre1 + (1 - alpha) * padre2
            else:
                hijo = padre1.copy()

            for d in range(dimensiones):
                if np.random.random() < tasa_mutacion:
                    rango = limites_max[d] - limites_min[d]
                    hijo[d] += np.random.normal(0, rango * 0.05)

            hijo = np.clip(hijo, limites_min, limites_max)
            nueva_gen.append(hijo)

        poblacion = np.array(nueva_gen)

    print("END EA")
    return poblacion[0].copy(), trayectoria



iteraciones = 100
tamanio_poblacion = 60
tasa_mutacion = 0.15
tasa_cruce = 0.75


# Rastrigin 2D
rastrigin_max = 5.12
bounds_rastrigin_2d = [(-rastrigin_max, rastrigin_max)] * 2

parametros_iniciales = rng.random(2) * rastrigin_max * 2 - rastrigin_max

parametros, trayectoria = algoritmo_evolutivo(
    rastrigin,
    bounds_rastrigin_2d,
    punto_inicial=parametros_iniciales,
    iteraciones=iteraciones,
    tamanio_poblacion=tamanio_poblacion,
    tasa_mutacion=tasa_mutacion,
    tasa_cruce=tasa_cruce,
)

ax_rastrigin = crear_heatmap(
    rastrigin,
    cantidad_puntos=2000,
    min_x=-rastrigin_max,
    max_x=rastrigin_max,
    min_y=-rastrigin_max,
    max_y=rastrigin_max,
)
crear_animacion(ax_rastrigin, rastrigin, trayectoria, "rastrigin_ea")

#  Rastrigin 3D
bounds_rastrigin_3d = [(-rastrigin_max, rastrigin_max)] * 3

parametros_iniciales_3d = rng.random(3) * rastrigin_max * 2 - rastrigin_max

parametros_3d, trayectoria_3d = algoritmo_evolutivo(
    rastrigin,
    bounds_rastrigin_3d,
    punto_inicial=parametros_iniciales_3d,
    iteraciones=iteraciones,
    tamanio_poblacion=tamanio_poblacion,
    tasa_mutacion=tasa_mutacion,
    tasa_cruce=tasa_cruce,
)

ax_rastrigin_3d = heatmap_3d(
    rastrigin,
    cantidad_puntos=30,
    min_x=-rastrigin_max,
    max_x=rastrigin_max,
    min_y=-rastrigin_max,
    max_y=rastrigin_max,
    min_z=-rastrigin_max,
    max_z=rastrigin_max,
)
crear_animacion_3d(ax_rastrigin_3d, rastrigin, trayectoria_3d, "rastrigin_3d_ea")


#  Rosenbrock 2D
rosenbrock_max = 2
bounds_rosen_2d = [(-rosenbrock_max, rosenbrock_max)] * 2

parametros_iniciales_rosen = rng.random(2) * rosenbrock_max * 2 - rosenbrock_max

parametros_rosen, trayectoria_rosen = algoritmo_evolutivo(
    rosenbrock,
    bounds_rosen_2d,
    punto_inicial=parametros_iniciales_rosen,
    iteraciones=iteraciones,
    tamanio_poblacion=tamanio_poblacion,
    tasa_mutacion=tasa_mutacion,
    tasa_cruce=tasa_cruce,
)

ax_rosen = crear_heatmap(
    rosenbrock,
    cantidad_puntos=2000,
    min_x=-rosenbrock_max,
    max_x=rosenbrock_max,
    min_y=-rosenbrock_max,
    max_y=rosenbrock_max,
)
crear_animacion(ax_rosen, rosenbrock, trayectoria_rosen, "rosenbrock_ea")


#  Rosenbrock 3D
bounds_rosen_3d = [(-rosenbrock_max, rosenbrock_max)] * 3

parametros_iniciales_rosen_3d = rng.random(3) * rosenbrock_max * 2 - rosenbrock_max

parametros_rosen_3d, trayectoria_rosen_3d = algoritmo_evolutivo(
    rosenbrock,
    bounds_rosen_3d,
    punto_inicial=parametros_iniciales_rosen_3d,
    iteraciones=iteraciones,
    tamanio_poblacion=tamanio_poblacion,
    tasa_mutacion=tasa_mutacion,
    tasa_cruce=tasa_cruce,
)

ax_rosen_3d = heatmap_3d(
    rosenbrock,
    cantidad_puntos=30,
    min_x=-rosenbrock_max,
    max_x=rosenbrock_max,
    min_y=-rosenbrock_max,
    max_y=rosenbrock_max,
    min_z=-rosenbrock_max,
    max_z=rosenbrock_max,
)
crear_animacion_3d(ax_rosen_3d, rosenbrock, trayectoria_rosen_3d, "rosenbrock_3d_ea")
