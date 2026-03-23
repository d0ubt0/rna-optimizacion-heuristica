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


def particle_swarm_optimization(
    funcion,
    dimensiones,
    min_val,
    max_val,
    num_particulas=30,
    iteraciones=100,
    w=0.7,
    c1=1.5,
    c2=1.5,
):
    posiciones = rng.uniform(min_val, max_val, (num_particulas, dimensiones))
    velocidades = rng.uniform(
        -(max_val - min_val),
         (max_val - min_val),
        (num_particulas, dimensiones)
    )

    mejor_personal = posiciones.copy()
    mejor_personal_valor = np.array([funcion(p) for p in posiciones])

    idx_mejor = np.argmin(mejor_personal_valor)
    mejor_global = mejor_personal[idx_mejor].copy()
    mejor_global_valor = mejor_personal_valor[idx_mejor]

    trayectoria = [mejor_global.copy()]
    trayectoria_enjambre = [posiciones.copy()]

    print("Start PSO")

    for iteracion in range(iteraciones):
        print(f"step: {iteracion}") if iteracion % 10 == 0 else 0

        r1 = rng.random((num_particulas, dimensiones))
        r2 = rng.random((num_particulas, dimensiones))

        velocidades = (
            w * velocidades
            + c1 * r1 * (mejor_personal - posiciones)
            + c2 * r2 * (mejor_global - posiciones)
        )

        posiciones = posiciones + velocidades
        posiciones = np.clip(posiciones, min_val, max_val)

        valores_actuales = np.array([funcion(p) for p in posiciones])
        mejoro = valores_actuales < mejor_personal_valor

        mejor_personal[mejoro] = posiciones[mejoro].copy()
        mejor_personal_valor[mejoro] = valores_actuales[mejoro]

        idx_mejor = np.argmin(mejor_personal_valor)
        if mejor_personal_valor[idx_mejor] < mejor_global_valor:
            mejor_global = mejor_personal[idx_mejor].copy()
            mejor_global_valor = mejor_personal_valor[idx_mejor]

        trayectoria.append(mejor_global.copy())
        trayectoria_enjambre.append(posiciones.copy())

    print("END PSO")
    return mejor_global, trayectoria, trayectoria_enjambre


iteraciones = 100
num_particulas = 30
w = 0.7
c1 = 1.5
c2 = 1.5

# Rastrigin 2D
rastrigin_max = 5.12

mejor_rastrigin, trayectoria_rastrigin, _ = particle_swarm_optimization(
    rastrigin,
    dimensiones=2,
    min_val=-rastrigin_max,
    max_val=rastrigin_max,
    num_particulas=num_particulas,
    iteraciones=iteraciones,
    w=w, c1=c1, c2=c2,
)

ax_rastrigin = crear_heatmap(
    rastrigin,
cantidad_puntos=2000,
min_x=-rastrigin_max,
max_x=rastrigin_max,
min_y=-rastrigin_max,
max_y=rastrigin_max
)
crear_animacion(ax_rastrigin, rastrigin, trayectoria_rastrigin, "rastrigin_pso")

# Rastrigin 3D (f: R³ → R)
mejor_rastrigin_3d, trayectoria_rastrigin_3d, _ = particle_swarm_optimization(
    rastrigin,
    dimensiones=3,
    min_val=-rastrigin_max,
    max_val=rastrigin_max,
    num_particulas=num_particulas,
    iteraciones=iteraciones,
    w=w, c1=c1, c2=c2,
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
crear_animacion_3d(ax_rastrigin_3d, rastrigin, trayectoria_rastrigin_3d, "rastrigin_3d_pso")

# Rosenbrock 2D
rosenbrock_max = 2

mejor_rosenbrock, trayectoria_rosenbrock, _ = particle_swarm_optimization(
    rosenbrock,
    dimensiones=2,
    min_val=-rosenbrock_max,
    max_val=rosenbrock_max,
    num_particulas=num_particulas,
    iteraciones=iteraciones,
    w=w, c1=c1, c2=c2,
)

ax_rosen = crear_heatmap(
    rosenbrock,
cantidad_puntos=2000,
min_x=-rosenbrock_max,
max_x=rosenbrock_max,
min_y=-rosenbrock_max,
max_y=rosenbrock_max
)
crear_animacion(ax_rosen, rosenbrock, trayectoria_rosenbrock, "rosenbrock_pso")

# Rosenbrock 3D (f: R³ → R)
mejor_rosenbrock_3d, trayectoria_rosenbrock_3d, _ = particle_swarm_optimization(
    rosenbrock,
    dimensiones=3,
    min_val=-rosenbrock_max,
    max_val=rosenbrock_max,
    num_particulas=num_particulas,
    iteraciones=iteraciones,
    w=w, c1=c1, c2=c2,
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
crear_animacion_3d(ax_rosen_3d, rosenbrock, trayectoria_rosenbrock_3d, "rosenbrock_3d_pso")
