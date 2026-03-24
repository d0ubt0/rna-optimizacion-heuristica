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


def evolucion_diferencial(
    funcion,
    dimension,
    min_val,
    max_val,
    poblacion_size=20,
    iteraciones=100,
    F=0.8,
    CR=0.9,
):
    poblacion = rng.random((poblacion_size, dimension)) * (max_val - min_val) + min_val
    fitness = np.array([funcion(ind) for ind in poblacion])
    mejor_idx = np.argmin(fitness)
    trayectoria = [poblacion[mejor_idx].copy()]

    print("Start DE")

    for gen in range(iteraciones):
        print(f"step: {gen}") if gen % 10 == 0 else 0

        for i in range(poblacion_size):
            indices = [idx for idx in range(poblacion_size) if idx != i]
            a, b, c = rng.choice(indices, size=3, replace=False)

            mutante = poblacion[a] + F * (poblacion[b] - poblacion[c])
            mutante = np.clip(mutante, min_val, max_val)

            mascara_cruce = rng.random(dimension) < CR
            j_rand = rng.integers(0, dimension)
            mascara_cruce[j_rand] = True

            trial = np.where(mascara_cruce, mutante, poblacion[i])

            fitness_trial = funcion(trial)
            if fitness_trial <= fitness[i]:
                poblacion[i] = trial
                fitness[i] = fitness_trial

        mejor_idx = np.argmin(fitness)
        trayectoria.append(poblacion[mejor_idx].copy())

    print("END DE")
    return poblacion[mejor_idx], trayectoria


iteraciones = 100
poblacion_size = 20
F = 0.8
CR = 0.9

# Rastrigin 2D
rastrigin_max = 5.12

mejor_rastrigin, trayectoria_rastrigin = evolucion_diferencial(
    funcion=rastrigin,
    dimension=2,
    min_val=-rastrigin_max,
    max_val=rastrigin_max,
    poblacion_size=poblacion_size,
    iteraciones=iteraciones,
    F=F,
    CR=CR,
)

ax_rastrigin = crear_heatmap(
    rastrigin,
    cantidad_puntos=2000,
    min_x=-rastrigin_max,
    max_x=rastrigin_max,
    min_y=-rastrigin_max,
    max_y=rastrigin_max,
)
crear_animacion(ax_rastrigin, rastrigin, trayectoria_rastrigin, "rastrigin_de")

# Rastrigin 3D (f: R³ → R)
mejor_rastrigin_3d, trayectoria_rastrigin_3d = evolucion_diferencial(
    funcion=rastrigin,
    dimension=3,
    min_val=-rastrigin_max,
    max_val=rastrigin_max,
    poblacion_size=poblacion_size,
    iteraciones=iteraciones,
    F=F,
    CR=CR,
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
crear_animacion_3d(ax_rastrigin_3d, rastrigin, trayectoria_rastrigin_3d, "rastrigin_3d_de")

# Rosenbrock 2D
rosenbrock_max = 2

mejor_rosenbrock, trayectoria_rosenbrock = evolucion_diferencial(
    funcion=rosenbrock,
    dimension=2,
    min_val=-rosenbrock_max,
    max_val=rosenbrock_max,
    poblacion_size=poblacion_size,
    iteraciones=iteraciones,
    F=F,
    CR=CR,
)

ax_rosen = crear_heatmap(
    rosenbrock,
    cantidad_puntos=2000,
    min_x=-rosenbrock_max,
    max_x=rosenbrock_max,
    min_y=-rosenbrock_max,
    max_y=rosenbrock_max,
)
crear_animacion(ax_rosen, rosenbrock, trayectoria_rosenbrock, "rosenbrock_de")

# Rosenbrock 3D (f: R³ → R)
mejor_rosenbrock_3d, trayectoria_rosenbrock_3d = evolucion_diferencial(
    funcion=rosenbrock,
    dimension=3,
    min_val=-rosenbrock_max,
    max_val=rosenbrock_max,
    poblacion_size=poblacion_size,
    iteraciones=iteraciones,
    F=F,
    CR=CR,
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
crear_animacion_3d(ax_rosen_3d, rosenbrock, trayectoria_rosenbrock_3d, "rosenbrock_3d_de")
