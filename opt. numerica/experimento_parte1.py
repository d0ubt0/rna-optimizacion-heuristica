import csv
from pathlib import Path

import numpy as np


# Benchmark functions and gradients (aligned with project definitions)
def rastrigin(x, A=10.0):
    x = np.asarray(x)
    d = x.size
    return float(A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def rastrigin_gradiente(x, A=10.0):
    x = np.asarray(x)
    return 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)


def rosenbrock(x):
    x = np.asarray(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def rosenbrock_gradiente(x):
    x = np.asarray(x)
    d = x.size
    g = np.zeros_like(x)
    g[0] = -400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1)
    for i in range(1, d - 1):
        g[i] = (
            -400 * x[i] * (x[i + 1] - x[i] ** 2)
            + 2 * (x[i] - 1)
            + 200 * (x[i] - x[i - 1] ** 2)
        )
    g[-1] = 200 * (x[-1] - x[-2] ** 2)
    return g


def descenso_por_gradiente(grad_fn, x0, iteraciones=100, tasa_aprendizaje=0.001):
    x = x0.copy()
    for _ in range(iteraciones):
        g = grad_fn(x)
        x = x - tasa_aprendizaje * g
    return x


def evolucion_diferencial(
    funcion,
    dimension,
    min_val,
    max_val,
    rng,
    poblacion_size=20,
    iteraciones=100,
    F=0.8,
    CR=0.9,
):
    poblacion = rng.random((poblacion_size, dimension)) * (max_val - min_val) + min_val
    fitness = np.array([funcion(ind) for ind in poblacion])

    for _ in range(iteraciones):
        for i in range(poblacion_size):
            indices = [idx for idx in range(poblacion_size) if idx != i]
            a, b, c = rng.choice(indices, size=3, replace=False)

            mutante = poblacion[a] + F * (poblacion[b] - poblacion[c])
            mutante = np.clip(mutante, min_val, max_val)

            mascara = rng.random(dimension) < CR
            mascara[rng.integers(0, dimension)] = True
            trial = np.where(mascara, mutante, poblacion[i])

            f_trial = funcion(trial)
            if f_trial <= fitness[i]:
                poblacion[i] = trial
                fitness[i] = f_trial

    return poblacion[int(np.argmin(fitness))]


def algoritmo_evolutivo(
    funcion_objetivo,
    bounds,
    punto_inicial,
    rng,
    iteraciones=100,
    tamanio_poblacion=50,
    tasa_mutacion=0.1,
    tasa_cruce=0.7,
    elitismo=2,
):
    dimensiones = len(bounds)
    limites_min = np.array([b[0] for b in bounds])
    limites_max = np.array([b[1] for b in bounds])

    poblacion = limites_min + rng.random((tamanio_poblacion, dimensiones)) * (
        limites_max - limites_min
    )
    poblacion[0] = np.clip(punto_inicial, limites_min, limites_max)

    for _ in range(iteraciones):
        fitness = np.array([funcion_objetivo(ind) for ind in poblacion])
        orden = np.argsort(fitness)
        poblacion = poblacion[orden]
        fitness = fitness[orden]

        nueva_gen = [poblacion[i].copy() for i in range(elitismo)]

        while len(nueva_gen) < tamanio_poblacion:
            def torneo(k=3):
                comp = rng.choice(tamanio_poblacion, k, replace=False)
                ganador = comp[np.argmin(fitness[comp])]
                return poblacion[ganador].copy()

            padre1 = torneo()
            padre2 = torneo()

            if rng.random() < tasa_cruce:
                alpha = rng.random(dimensiones)
                hijo = alpha * padre1 + (1 - alpha) * padre2
            else:
                hijo = padre1.copy()

            for d in range(dimensiones):
                if rng.random() < tasa_mutacion:
                    rango = limites_max[d] - limites_min[d]
                    hijo[d] += rng.normal(0, rango * 0.05)

            hijo = np.clip(hijo, limites_min, limites_max)
            nueva_gen.append(hijo)

        poblacion = np.array(nueva_gen)

    fitness_final = np.array([funcion_objetivo(ind) for ind in poblacion])
    return poblacion[int(np.argmin(fitness_final))]


def particle_swarm_optimization(
    funcion,
    dimensiones,
    min_val,
    max_val,
    rng,
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
        (num_particulas, dimensiones),
    )

    mejor_personal = posiciones.copy()
    mejor_personal_valor = np.array([funcion(p) for p in posiciones])

    idx_mejor = int(np.argmin(mejor_personal_valor))
    mejor_global = mejor_personal[idx_mejor].copy()
    mejor_global_valor = mejor_personal_valor[idx_mejor]

    for _ in range(iteraciones):
        r1 = rng.random((num_particulas, dimensiones))
        r2 = rng.random((num_particulas, dimensiones))

        velocidades = (
            w * velocidades
            + c1 * r1 * (mejor_personal - posiciones)
            + c2 * r2 * (mejor_global - posiciones)
        )

        posiciones = posiciones + velocidades
        posiciones = np.clip(posiciones, min_val, max_val)

        valores = np.array([funcion(p) for p in posiciones])
        mejoro = valores < mejor_personal_valor

        mejor_personal[mejoro] = posiciones[mejoro].copy()
        mejor_personal_valor[mejoro] = valores[mejoro]

        idx_mejor = int(np.argmin(mejor_personal_valor))
        if mejor_personal_valor[idx_mejor] < mejor_global_valor:
            mejor_global = mejor_personal[idx_mejor].copy()
            mejor_global_valor = mejor_personal_valor[idx_mejor]

    return mejor_global


def evaluaciones_por_run(metodo, iteraciones=100, pop_de=20, pop_ea=50, pop_pso=30):
    if metodo == 'GD':
        return iteraciones + 1
    if metodo == 'DE':
        return pop_de + iteraciones * pop_de + 1
    if metodo == 'EA':
        return iteraciones * pop_ea + 1
    if metodo == 'PSO':
        return pop_pso + iteraciones * pop_pso + 1
    raise ValueError(f'Metodo no reconocido: {metodo}')


def ejecutar_caso(funcion_nombre, dimension, metodo, corridas=20):
    if funcion_nombre == 'Rastrigin':
        funcion = rastrigin
        gradiente = rastrigin_gradiente
        dominio_max = 5.12
    elif funcion_nombre == 'Rosenbrock':
        funcion = rosenbrock
        gradiente = rosenbrock_gradiente
        dominio_max = 2.0
    else:
        raise ValueError(f'Funcion no soportada: {funcion_nombre}')

    valores_finales = []
    for semilla in range(corridas):
        rng = np.random.default_rng(semilla)

        if metodo == 'GD':
            x0 = rng.random(dimension) * dominio_max * 2 - dominio_max
            xf = descenso_por_gradiente(
                gradiente,
                x0,
                iteraciones=100,
                tasa_aprendizaje=0.001,
            )
        elif metodo == 'DE':
            xf = evolucion_diferencial(
                funcion,
                dimension,
                -dominio_max,
                dominio_max,
                rng,
                poblacion_size=20,
                iteraciones=100,
                F=0.8,
                CR=0.9,
            )
        elif metodo == 'EA':
            bounds = [(-dominio_max, dominio_max)] * dimension
            x0 = rng.random(dimension) * dominio_max * 2 - dominio_max
            xf = algoritmo_evolutivo(
                funcion,
                bounds,
                x0,
                rng,
                iteraciones=100,
                tamanio_poblacion=50,
                tasa_mutacion=0.1,
                tasa_cruce=0.7,
                elitismo=2,
            )
        elif metodo == 'PSO':
            xf = particle_swarm_optimization(
                funcion,
                dimension,
                -dominio_max,
                dominio_max,
                rng,
                num_particulas=30,
                iteraciones=100,
                w=0.7,
                c1=1.5,
                c2=1.5,
            )
        else:
            raise ValueError(f'Metodo no soportado: {metodo}')

        valores_finales.append(funcion(xf))

    arr = np.array(valores_finales, dtype=float)
    return {
        'funcion': funcion_nombre,
        'dimension': dimension,
        'metodo': metodo,
        'corridas': corridas,
        'valor_objetivo_promedio': float(np.mean(arr)),
        'desviacion': float(np.std(arr)),
        'mejor_valor': float(np.min(arr)),
        'peor_valor': float(np.max(arr)),
        'evaluaciones_promedio': evaluaciones_por_run(metodo),
    }


def main():
    rows = []
    for funcion in ('Rastrigin', 'Rosenbrock'):
        for dimension in (2, 3):
            for metodo in ('GD', 'DE', 'EA', 'PSO'):
                rows.append(ejecutar_caso(funcion, dimension, metodo, corridas=20))

    out_path = Path('opt. numerica') / 'resultados_parte1_20corridas.csv'
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'funcion',
                'dimension',
                'metodo',
                'corridas',
                'valor_objetivo_promedio',
                'desviacion',
                'mejor_valor',
                'peor_valor',
                'evaluaciones_promedio',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'Se escribio: {out_path}')


if __name__ == '__main__':
    main()
