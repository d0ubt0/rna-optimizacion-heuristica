import csv
import time

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

# ─────────────────────────────────────────────────────────────────────────────
#  Función principal de prueba
# ─────────────────────────────────────────────────────────────────────────────

def modo_test(algoritmo, funcion, dim, dominio_max, n_ejecuciones=10, **kwargs):
    """
    Ejecuta un algoritmo de optimización n_ejecuciones veces con semillas
    distintas y devuelve un diccionario con métricas estadísticas del
    valor final de la función objetivo.

    Parámetros
    ----------
    algoritmo     : str   – "gradiente" | "diferencial" | "evolutivo" | "particulas"
    funcion       : callable
    dim           : int   – dimensión del problema (2 ó 3)
    dominio_max   : float – límite del dominio (±dominio_max)
    n_ejecuciones : int   – cantidad de semillas / runs (default 10)
    **kwargs      : parámetros extra que se pasan al algoritmo

    Retorna
    -------
    dict con las siguientes métricas:
        promedio, mediana, std, minimo, maximo,
        mejor_solucion (vector), tiempo_total_s,
        tasa_exito (fracción de runs que llegan a < umbral),
        iqr (rango intercuartil), coef_variacion
    """
    valores_finales = []
    mejores_soluciones = []
    tiempos = []

    for semilla in range(n_ejecuciones):
        rng = np.random.default_rng(semilla)
        min_val = -dominio_max
        max_val =  dominio_max

        t0 = time.perf_counter()

        # ── Descenso por gradiente ───────────────────────────────
        if algoritmo == "gradiente":
            gradiente_fn     = kwargs.get("gradiente_funcion")
            iteraciones      = kwargs.get("iteraciones", 100)
            tasa_aprendizaje = kwargs.get("tasa_de_aprendizaje", 0.001)

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
            bounds    = [(min_val, max_val)] * dim
            punto_ini = rng.random(dim) * dominio_max * 2 - dominio_max
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
            raise ValueError(
                f"Algoritmo desconocido: '{algoritmo}'. "
                "Opciones: 'gradiente', 'diferencial', 'evolutivo', 'particulas'"
            )

        t1 = time.perf_counter()
        tiempos.append(t1 - t0)

        val = funcion(params_finales)
        valores_finales.append(val)
        mejores_soluciones.append((val, params_finales.copy()))

    arr = np.array(valores_finales)
    mejor_val, mejor_vec = min(mejores_soluciones, key=lambda x: x[0])

    # Umbral de éxito: valor < 0.01  (ajustar si se desea)
    umbral_exito = kwargs.get("umbral_exito", 0.01)
    tasa_exito   = float(np.mean(arr < umbral_exito))

    q25, q75 = np.percentile(arr, [25, 75])
    cv = float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else float("nan")

    return {
        "promedio":        float(np.mean(arr)),
        "mediana":         float(np.median(arr)),
        "std":             float(np.std(arr)),
        "minimo":          float(np.min(arr)),
        "maximo":          float(np.max(arr)),
        "iqr":             float(q75 - q25),
        "coef_variacion":  cv,
        "tasa_exito":      tasa_exito,
        "tiempo_total_s":  float(np.sum(tiempos)),
        "tiempo_medio_s":  float(np.mean(tiempos)),
        "mejor_solucion":  mejor_vec,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Exportar a CSV
# ─────────────────────────────────────────────────────────────────────────────

def exportar_csv(resultados: dict, ruta = None) -> str:
    """
    Guarda el diccionario de resultados en un archivo CSV.

    Parámetros
    ----------
    resultados : dict[str, dict]  – salida de modo_test para cada experimento
    ruta       : str | None       – ruta del CSV (None = genera nombre con timestamp)

    Retorna
    -------
    str – ruta del archivo generado
    """
    if ruta is None:
        ruta = "opt. numerica/resultados.csv"

    metricas_escalares = [
        "promedio", "mediana", "std", "minimo", "maximo",
        "iqr", "coef_variacion", "tasa_exito",
        "tiempo_total_s", "tiempo_medio_s",
    ]

    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Encabezado
        writer.writerow(["experimento"] + metricas_escalares + ["mejor_solucion"])

        for nombre, metricas in resultados.items():
            fila = [nombre]
            for m in metricas_escalares:
                val = metricas.get(m, "")
                fila.append(f"{val:.8f}" if isinstance(val, float) else val)

            # Mejor solución como cadena separada por "|"
            mejor = metricas.get("mejor_solucion", [])
            fila.append("|".join(f"{x:.8f}" for x in mejor))

            writer.writerow(fila)

    return ruta


# ─────────────────────────────────────────────────────────────────────────────
#  Imprimir tabla en consola
# ─────────────────────────────────────────────────────────────────────────────

def imprimir_tabla(resultados: dict) -> None:
    col_exp   = 24
    col_num   = 11

    cabeceras = [
        "Experimento",
        "Promedio", "Mediana", "Std", "Mínimo", "Máximo",
        "IQR", "CV", "Éxito%", "t̄ (s)",
    ]

    sep = "─" * (col_exp + col_num * (len(cabeceras) - 1) + len(cabeceras) - 1)
    print(f"\n{sep}")
    print(
        f"{'Experimento':<{col_exp}}"
        + "".join(f"{h:>{col_num}}" for h in cabeceras[1:])
    )
    print(sep)

    for nombre, m in resultados.items():
        print(
            f"{nombre:<{col_exp}}"
            f"{m['promedio']:>{col_num}.5f}"
            f"{m['mediana']:>{col_num}.5f}"
            f"{m['std']:>{col_num}.5f}"
            f"{m['minimo']:>{col_num}.5f}"
            f"{m['maximo']:>{col_num}.5f}"
            f"{m['iqr']:>{col_num}.5f}"
            f"{m['coef_variacion']:>{col_num}.4f}"
            f"{m['tasa_exito']*100:>{col_num}.1f}"
            f"{m['tiempo_medio_s']:>{col_num}.4f}"
        )

    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Ejemplo de uso
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    resultados = {
        # Descenso por gradiente
        "GD  Rastrigin 2D":  modo_test("gradiente",   rastrigin,  dim=2, dominio_max=5.12,
                                        gradiente_funcion=rastrigin_gradiente),
        "GD  Rastrigin 3D":  modo_test("gradiente",   rastrigin,  dim=3, dominio_max=5.12,
                                        gradiente_funcion=rastrigin_gradiente),
        "GD  Rosenbrock 2D": modo_test("gradiente",   rosenbrock, dim=2, dominio_max=2.0,
                                        gradiente_funcion=rosenbrock_gradiente),
        "GD  Rosenbrock 3D": modo_test("gradiente",   rosenbrock, dim=3, dominio_max=2.0,
                                        gradiente_funcion=rosenbrock_gradiente),
        # Evolución diferencial
        "DE  Rastrigin 2D":  modo_test("diferencial", rastrigin,  dim=2, dominio_max=5.12),
        "DE  Rastrigin 3D":  modo_test("diferencial", rastrigin,  dim=3, dominio_max=5.12),
        "DE  Rosenbrock 2D": modo_test("diferencial", rosenbrock, dim=2, dominio_max=2.0),
        "DE  Rosenbrock 3D": modo_test("diferencial", rosenbrock, dim=3, dominio_max=2.0),
        # Algoritmo evolutivo
        "EA  Rastrigin 2D":  modo_test("evolutivo",   rastrigin,  dim=2, dominio_max=5.12),
        "EA  Rastrigin 3D":  modo_test("evolutivo",   rastrigin,  dim=3, dominio_max=5.12),
        "EA  Rosenbrock 2D": modo_test("evolutivo",   rosenbrock, dim=2, dominio_max=2.0),
        "EA  Rosenbrock 3D": modo_test("evolutivo",   rosenbrock, dim=3, dominio_max=2.0),
        # Enjambre de partículas
        "PSO Rastrigin 2D":  modo_test("particulas",  rastrigin,  dim=2, dominio_max=5.12),
        "PSO Rastrigin 3D":  modo_test("particulas",  rastrigin,  dim=3, dominio_max=5.12),
        "PSO Rosenbrock 2D": modo_test("particulas",  rosenbrock, dim=2, dominio_max=2.0),
        "PSO Rosenbrock 3D": modo_test("particulas",  rosenbrock, dim=3, dominio_max=2.0),
    }

    # Imprimir tabla en consola
    imprimir_tabla(resultados)

    # Guardar CSV
    ruta_csv = exportar_csv(resultados)
    print(f"CSV guardado en: {ruta_csv}\n")
