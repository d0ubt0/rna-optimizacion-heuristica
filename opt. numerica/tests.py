import numpy as np
from evolucion_diferencial import evolucion_diferencial
from evolutivo import algoritmo_evolutivo
from funciones import (
    rastrigin,
    rastrigin_gradiente,
    rosenbrock,
    rosenbrock_gradiente,
)
from gradiente import descenso_por_gradiente
from particulas import particle_swarm_optimization


def modo_test(algoritmo, funcion, dim, dominio_max, n_ejecuciones=10, **kwargs):
    """
    Ejecuta un algoritmo de optimización n_ejecuciones veces con semillas
    distintas y devuelve el promedio del valor final de la función.

    Parámetros
    ----------
    algoritmo     : str   – "gradiente" | "diferencial" | "evolutivo" | "particulas"
    funcion       : callable
    dim           : int   – dimensión del problema (2 ó 3)
    dominio_max   : float – límite del dominio (±dominio_max)
    n_ejecuciones : int   – cantidad de semillas / runs (default 10)
    **kwargs      : parámetros extra que se pasan al algoritmo
                    (ej. iteraciones, tasa_de_aprendizaje, F, CR, …)

    Retorna
    -------
    float – promedio del valor de la función en el punto final
    """
    valores_finales = []

    for semilla in range(n_ejecuciones):
        rng = np.random.default_rng(semilla)
        min_val = -dominio_max
        max_val =  dominio_max

        # ── Descenso por gradiente ───────────────────────────────
        if algoritmo == "gradiente":
            gradiente_fn      = kwargs.get("gradiente_funcion")
            iteraciones       = kwargs.get("iteraciones", 100)
            tasa_aprendizaje  = kwargs.get("tasa_de_aprendizaje", 0.001)

            params_ini = rng.random(dim) * dominio_max * 2 - dominio_max
            params_finales, _ = descenso_por_gradiente(
                gradiente_fn, params_ini, iteraciones, tasa_aprendizaje
            )

        # ── Evolución diferencial ────────────────────────────────
        elif algoritmo == "diferencial":
            params_finales, _ = evolucion_diferencial(
                funcion=funcion,
                dimension=dim,
                min_val=min_val,
                max_val=max_val,
                poblacion_size=kwargs.get("poblacion_size", 20),
                iteraciones=kwargs.get("iteraciones", 100),
                F=kwargs.get("F", 0.8),
                CR=kwargs.get("CR", 0.9),
            )

        # ── Algoritmo evolutivo ──────────────────────────────────
        elif algoritmo == "evolutivo":
            bounds     = [(min_val, max_val)] * dim
            punto_ini  = rng.random(dim) * dominio_max * 2 - dominio_max
            params_finales, _ = algoritmo_evolutivo(
                funcion_objetivo=funcion,
                bounds=bounds,
                punto_inicial=punto_ini,
                iteraciones=kwargs.get("iteraciones", 100),
                tamanio_poblacion=kwargs.get("tamanio_poblacion", 50),
                tasa_mutacion=kwargs.get("tasa_mutacion", 0.1),
                tasa_cruce=kwargs.get("tasa_cruce", 0.7),
                elitismo=kwargs.get("elitismo", 2),
            )

        # ── Enjambre de partículas ───────────────────────────────
        elif algoritmo == "particulas":
            params_finales, _, _ = particle_swarm_optimization(
                funcion=funcion,
                dimensiones=dim,
                min_val=min_val,
                max_val=max_val,
                num_particulas=kwargs.get("num_particulas", 30),
                iteraciones=kwargs.get("iteraciones", 100),
                w=kwargs.get("w", 0.7),
                c1=kwargs.get("c1", 1.5),
                c2=kwargs.get("c2", 1.5),
            )

        else:
            raise ValueError(f"Algoritmo desconocido: '{algoritmo}'. "
                             "Opciones: 'gradiente', 'diferencial', 'evolutivo', 'particulas'")

        valores_finales.append(funcion(params_finales))

    return float(np.mean(valores_finales))


# ──────────────────────────────────────────────
#  Ejemplo de uso
# ──────────────────────────────────────────────

if __name__ == "__main__":
    resultados = {
        # Gradiente
        "GD  Rastrigin 2D": modo_test("gradiente",   rastrigin,  dim=2, dominio_max=5.12,
                                       gradiente_funcion=rastrigin_gradiente),
        "GD  Rastrigin 3D": modo_test("gradiente",   rastrigin,  dim=3, dominio_max=5.12,
                                       gradiente_funcion=rastrigin_gradiente),
        "GD  Rosenbrock 2D": modo_test("gradiente",  rosenbrock, dim=2, dominio_max=2.0,
                                        gradiente_funcion=rosenbrock_gradiente),
        "GD  Rosenbrock 3D": modo_test("gradiente",  rosenbrock, dim=3, dominio_max=2.0,
                                        gradiente_funcion=rosenbrock_gradiente),
        # Evolución diferencial
        "DE  Rastrigin 2D":  modo_test("diferencial", rastrigin,  dim=2, dominio_max=5.12),
        "DE  Rastrigin 3D":  modo_test("diferencial", rastrigin,  dim=3, dominio_max=5.12),
        "DE  Rosenbrock 2D": modo_test("diferencial", rosenbrock, dim=2, dominio_max=2.0),
        "DE  Rosenbrock 3D": modo_test("diferencial", rosenbrock, dim=3, dominio_max=2.0),
        # Evolutivo
        "EA  Rastrigin 2D":  modo_test("evolutivo",  rastrigin,  dim=2, dominio_max=5.12),
        "EA  Rastrigin 3D":  modo_test("evolutivo",  rastrigin,  dim=3, dominio_max=5.12),
        "EA  Rosenbrock 2D": modo_test("evolutivo",  rosenbrock, dim=2, dominio_max=2.0),
        "EA  Rosenbrock 3D": modo_test("evolutivo",  rosenbrock, dim=3, dominio_max=2.0),
        # Partículas
        "PSO Rastrigin 2D":  modo_test("particulas", rastrigin,  dim=2, dominio_max=5.12),
        "PSO Rastrigin 3D":  modo_test("particulas", rastrigin,  dim=3, dominio_max=5.12),
        "PSO Rosenbrock 2D": modo_test("particulas", rosenbrock, dim=2, dominio_max=2.0),
        "PSO Rosenbrock 3D": modo_test("particulas", rosenbrock, dim=3, dominio_max=2.0),
    }

    print(f"\n{'─'*42}")
    print(f"  {'Experimento':<22}  {'Promedio':>10}")
    print(f"{'─'*42}")
    for nombre, valor in resultados.items():
        print(f"  {nombre:<22}  {valor:>10.6f}")
    print(f"{'─'*42}\n")
