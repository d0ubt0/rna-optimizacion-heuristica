# Reporte técnico: Optimización numérica y combinatoria

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Equipo:** Sebastián Pabón Núñez, Jhofred Jahat Camacho Gómez  
**Fecha:** 24/03/2026  
**Repositorio de código:** [Github](https://github.com/d0ubt0/rna-optimizacion-heuristica)

## Resumen ejecutivo
Se resolvieron dos problemas: (1) optimización numérica en funciones de prueba continuas y (2) optimización combinatoria de rutas en las 32 capitales estatales de México. Para la Parte 1 se compararon descenso por gradiente, algoritmo evolutivo (EA), PSO y evolución diferencial (DE), en 2D y 3D, con 20 corridas por caso. Para la Parte 2 se compararon ACO y GA variando el valor-hora del vendedor, con costos de tiempo, peajes y combustible.

## 1) Parte 1: Optimización numérica

### 1.1 Funciones seleccionadas
De la lista propuesta se eligieron:
- **Función 1:** Rastrigin.
- **Función 2:** Rosenbrock.

Estas funciones son estándar en evaluación de optimizadores globales y locales porque presentan geometrías muy diferentes del paisaje objetivo (Jamil & Yang, 2013; Surjanovic & Bingham, 2013).

**Ecuación 1. Función de Rastrigin (n dimensiones)**

\[
f(\mathbf{x}) = 10n + \sum_{i=1}^{n}\left(x_i^2 - 10\cos(2\pi x_i)\right)
\]

Con dominio típico \(x_i \in [-5.12,5.12]\), mínimo global en \(\mathbf{x}=\mathbf{0}\) y \(f(\mathbf{0})=0\). Su alta multimodalidad la hace útil para evaluar exploración y riesgo de caer en mínimos locales (Surjanovic & Bingham, 2013; Jamil & Yang, 2013).

**Ecuación 2. Función de Rosenbrock (n dimensiones)**

\[
f(\mathbf{x}) = \sum_{i=1}^{n-1}\left[100(x_{i+1}-x_i^2)^2 + (1-x_i)^2\right]
\]

Tiene un valle estrecho y curvo con mínimo global en \(\mathbf{x}=(1,\ldots,1)\) y \(f(\mathbf{x})=0\). Es especialmente útil para evaluar estabilidad numérica y precisión de convergencia fina (Rosenbrock, 1960; Surjanovic & Bingham, 2013).

### 1.2 Métodos implementados
Se resolvió cada función en 2D y 3D con:
1. Descenso por gradiente (GD) con condición inicial aleatoria.
2. Algoritmo evolutivo (EA).
3. Optimización por enjambre de partículas (PSO).
4. Evolución diferencial (DE).

Estos métodos representan enfoques complementarios: un método basado en gradiente para explotación local y tres heurísticos poblacionales para exploración global (Nocedal & Wright, 2006; Holland, 1992; Kennedy & Eberhart, 1995; Storn & Price, 1997).

### 1.3 Configuración experimental
- Corridas por caso: **20** (semillas `0..19`).
- Dimensiones: **2D y 3D**.
- Iteraciones (todos): **100**.
- Parámetros GD: `lr=0.001`.
- Parámetros DE: `población=20`, `F=0.8`, `CR=0.9`.
- Parámetros EA: `población=50`, `mutación=0.1`, `cruce=0.7`, `elitismo=2`.
- Parámetros PSO: `partículas=30`, `w=0.7`, `c1=1.5`, `c2=1.5`.

**Reproducibilidad (Parte 1):**
```bash
python "opt. numerica/experimento_parte1.py"
```
Esto genera `opt. numerica/resultados_parte1_20corridas.csv`.

### 1.4 Métricas y resultados
Métricas reportadas:
- Valor final de la función objetivo (promedio, desviación y mejor caso).
- Número de evaluaciones promedio por corrida.

**Criterio de evaluaciones usado en este reporte:**
- GD: `iteraciones + 1 = 101`.
- DE: `población + iteraciones*población + 1 = 2021`.
- EA: `iteraciones*población + 1 = 5001`.
- PSO: `partículas + iteraciones*partículas + 1 = 3031`.

Para facilitar la lectura, primero se reportan resultados en Rastrigin (Tablas 1 y 2) y luego en Rosenbrock (Tablas 3 y 4), manteniendo el mismo orden de métodos.

#### Tabla 1. Rastrigin 2D (20 corridas)
| Método | Promedio f(x) | Desviación | Mejor | Evaluaciones |
|---|---:|---:|---:|---:|
| GD | 18.108137 | 9.982277 | 1.989918 | 101 |
| EA | 4.350e-05 | 8.693e-05 | 7.105e-15 | 5001 |
| PSO | 5.885e-07 | 2.517e-06 | 2.165e-11 | 3031 |
| DE | 0.099496 | 0.298488 | 7.105e-15 | 2021 |

La Tabla 1 muestra que, en Rastrigin 2D, los métodos heurísticos (EA/PSO/DE) alcanzan mejores valores finales que GD, aunque con mayor número de evaluaciones.

#### Tabla 2. Rastrigin 3D (20 corridas)
| Método | Promedio f(x) | Desviación | Mejor | Evaluaciones |
|---|---:|---:|---:|---:|
| GD | 26.067764 | 12.240703 | 4.974790 | 101 |
| EA | 7.814e-04 | 0.002294 | 7.015e-08 | 5001 |
| PSO | 0.151540 | 0.354368 | 2.003e-09 | 3031 |
| DE | 0.717205 | 0.600599 | 4.775e-05 | 2021 |

En la Tabla 2, al aumentar a 3D, se mantiene la ventaja de heurísticos sobre GD en calidad de solución, con incremento de variabilidad en PSO y DE.

#### Tabla 3. Rosenbrock 2D (20 corridas)
| Método | Promedio f(x) | Desviación | Mejor | Evaluaciones |
|---|---:|---:|---:|---:|
| GD | 1.294852 | 1.506695 | 0.001980 | 101 |
| EA | 0.061635 | 0.112428 | 2.932e-06 | 5001 |
| PSO | 5.848e-06 | 1.310e-05 | 4.531e-10 | 3031 |
| DE | 5.507e-15 | 1.019e-14 | 8.828e-19 | 2021 |

La Tabla 3 evidencia que, en Rosenbrock 2D, DE obtiene la mejor precisión promedio y mejor caso, consistente con su buen desempeño en valles curvos.

#### Tabla 4. Rosenbrock 3D (20 corridas)
| Método | Promedio f(x) | Desviación | Mejor | Evaluaciones |
|---|---:|---:|---:|---:|
| GD | 1.408352 | 1.021637 | 0.005781 | 101 |
| EA | 0.737961 | 0.572848 | 0.184525 | 5001 |
| PSO | 0.089464 | 0.047688 | 4.998e-05 | 3031 |
| DE | 0.002521 | 0.010925 | 8.549e-09 | 2021 |

En la Tabla 4, DE y PSO mantienen mejor calidad que GD y EA en 3D, aunque con el costo computacional esperado de métodos poblacionales.

Fuente de datos de tablas: `opt. numerica/resultados_parte1_20corridas.csv`.

### 1.5 Animaciones (obligatorio)
- **GIF gradiente:** [animacion_gd_rastrigin_2d.gif](opt.%20numerica/outputs/animacion_gd_rastrigin_2d.gif)
- **GIF heurístico (DE):** [animacion_de_rastrigin_2d.gif](opt.%20numerica/outputs/animacion_de_rastrigin_2d.gif)

Para ilustrar dinámicas de convergencia, a continuación se presenta primero GD y luego DE sobre la misma función objetivo.

**Figura 1.** Trayectoria de optimización de descenso por gradiente en Rastrigin 2D.

![Figura 1 - GD Rastrigin 2D](opt.%20numerica/outputs/animacion_gd_rastrigin_2d.gif)

La Figura 1 muestra una trayectoria más sensible a la inicialización y al valle local en esta función multimodal.

**Figura 2.** Trayectoria de optimización por evolución diferencial en Rastrigin 2D.

![Figura 2 - DE Rastrigin 2D](opt.%20numerica/outputs/animacion_de_rastrigin_2d.gif)

La Figura 2 muestra una exploración poblacional más robusta, coherente con los mejores valores finales observados en las tablas.

### 1.6 Discusión solicitada
- **Aporte de GD:** menor costo computacional por corrida (101 evaluaciones), implementación directa cuando existe gradiente analítico, y convergencia razonable en regiones suaves.
- **Aporte de heurísticos:** mejores mínimos finales en problemas multimodales y no convexos; en particular DE/PSO/EA superan ampliamente a GD en Rastrigin y Rosenbrock 3D.
- **Trade-off principal:** GD usa muchas menos evaluaciones, pero obtiene peores valores finales promedio en la mayoría de los casos de este estudio.
- **Necesidad de múltiples corridas:** sí. La variabilidad observada (desviaciones no nulas) confirma sensibilidad a semilla e inicialización en métodos heurísticos y también en GD por inicialización aleatoria.

## 2) Parte 2: Optimización combinatoria (TSP de 32 capitales)

### 2.1 Definición del problema
Un vendedor debe visitar todas las capitales de los 32 estados de México y regresar al origen (Ciudad de México en esta configuración).

### 2.2 Modelado de costo y trazabilidad de fuentes
El costo entre ciudades se modeló con una función compuesta de tiempo, peaje y combustible.

**Ecuación 3. Costo total por arco**
\[
C_{ij} = (valor\_hora \cdot tiempo_{ij}) + peajes_{ij} + combustible_{ij}
\]

**Ecuación 4. Costo de combustible por arco**

\[
combustible_{ij} = distancia_{ij} \cdot \frac{precio\_litro}{rendimiento\_{km/L}}
\]

La Ecuación 3 integra costo de oportunidad temporal y costos directos de viaje, mientras que la Ecuación 4 explicita el aporte energético del vehículo según rendimiento.

**Fuente y metodología de extracción (corrección solicitada):**
- Se dejó un pipeline ETL reproducible en `opt. combinatoria/scripts/costs_etl_inegi.py`.
- Snapshot de matriz base: `opt. combinatoria/data/costos_matriz.csv`.
- Metadatos de extracción: `opt. combinatoria/data/costos_fuentes.json`.
- Snapshot de trabajo bloqueado (rama feat/inegi-full-rerun): SHA-256 6774e6907f37819d70a891db3e3f148dc03409c1703b114505241f1824c9b60a (timestamp UTC 2026-04-09T02:41:24.213206+00:00).
- Endpoints oficiales usados de INEGI SAKBÉ v3.1:
  - `buscalinea` (ajuste de cada capital a la red carretera),
  - `optima` (distancia, tiempo y costo de peaje por par origen-destino),
  - `combustible` (precio de referencia por tipo de combustible).
- Token usado por variable de entorno (`INEGI_RUTEO_TOKEN`), sin exponer credenciales en el repositorio.

Estas fuentes se seleccionaron por su carácter oficial y cobertura nacional para red vial y costos asociados (INEGI, 2026; CAPUFE, 2025; CNE, 2025).

**Comando de actualización del snapshot:**
```bash
$env:INEGI_RUTEO_TOKEN="<tu_token>"
python "opt. combinatoria/scripts/costs_etl_inegi.py" --ignore-existing
```

**Contingencia manual (si la API no está disponible):**
1. Mantener `costos_matriz.csv` como snapshot congelado.
2. Recolectar distancia/tiempo/peaje por pares desde fuentes oficiales (INEGI Ruteo o SICT/CAPUFE).
3. Documentar fecha, fuente y criterio de captura en `costos_fuentes.json`.
4. Validar matriz completa 32x32 antes de ejecutar ACO/GA.

### 2.3 Vehículo y parámetro estudiado
- Vehículo seleccionado: **Sedan Gasolina** (`vehicle_id=sedan_gasolina`).
- Rendimiento: **15.5 km/L**.
- Precio combustible configurado: **23.415 MXN/L** (referencia INEGI combustible tipo Regular).
- Parámetro analizado: `valor_hora` en **[100, 300] MXN/h** con paso **50**.

### 2.4 Métodos implementados
- Colonia de hormigas (ACO).
- Algoritmo genético (GA).

Configuración usada (archivo `opt. combinatoria/data/config.yaml`):
- Seeds: `[7, 13, 29]`
- ACO: `num_ants=55`, `iterations=120`, `alpha=1.0`, `beta=3.0`, `evaporation=0.35`, `q=120.0`.
- GA: `population_size=140`, `generations=220`, `crossover_rate=0.9`, `mutation_rate=0.22`, `elite_size=4`, `tournament_size=4`.

### 2.5 Resultados y visualización (mapa real)
En esta sección se presenta primero la comparación cuantitativa ACO vs GA y luego la evidencia visual del comportamiento del GA sobre mapa real.

#### Tabla 5. Comparativa ACO vs GA por valor-hora
| Valor hora (MXN/h) | Algoritmo | Mejor costo (MXN) | Promedio (MXN) | Desviación | Seed mejor |
|---:|---|---:|---:|---:|---:|
| 100 | ACO | 46698.84 | 47248.37 | 397.25 | 7 |
| 100 | GA | 51931.08 | 53497.88 | 1957.28 | 29 |
| 150 | ACO | 57038.91 | 57416.31 | 297.03 | 29 |
| 150 | GA | 60204.85 | 62179.38 | 1742.49 | 29 |
| 200 | ACO | 66286.27 | 66483.09 | 170.13 | 13 |
| 200 | GA | 69546.06 | 70632.68 | 907.06 | 13 |
| 250 | ACO | 75035.99 | 75788.82 | 542.52 | 13 |
| 250 | GA | 74821.81 | 79040.76 | 3051.05 | 7 |
| 300 | ACO | 83556.68 | 85088.15 | 1101.31 | 29 |
| 300 | GA | 86982.29 | 91954.07 | 3640.59 | 29 |

La Tabla 5 indica que ACO ofrece el mejor desempeño global en este experimento, mientras GA solo supera el mejor-caso de ACO en `valor_hora=250` y con mayor dispersión.

**Mejor solución global observada:** ACO con `valor_hora=100`, costo `46,698.84 MXN`.

Artefactos:
- **GIF de iteraciones GA sobre mapa real:** [ruta_ga_iteraciones_mapa_real.gif](opt.%20combinatoria/outputs/ruta_ga_iteraciones_mapa_real.gif)
- **Figura final GA sobre mapa real:** [ruta_ga_iteraciones_mapa_real.png](opt.%20combinatoria/outputs/ruta_ga_iteraciones_mapa_real.png)
- **GIF mejor solución global:** [mejor_ruta_global.gif](opt.%20combinatoria/outputs/mejor_ruta_global.gif)
- **Figura final mejor solución global:** [mejor_ruta_global.png](opt.%20combinatoria/outputs/mejor_ruta_global.png)
- **Rutas por método:** [mejor_ruta_aco.csv](opt.%20combinatoria/outputs/mejor_ruta_aco.csv), [mejor_ruta_ga.csv](opt.%20combinatoria/outputs/mejor_ruta_ga.csv)

**Figura 3.** Iteraciones del algoritmo genético sobre mapa real de México.

![Figura 3 - Iteraciones GA en mapa real](opt.%20combinatoria/outputs/ruta_ga_iteraciones_mapa_real.gif)

La Figura 3 permite verificar visualmente la evolución iterativa del GA sobre cartografía real, cumpliendo el requisito de seguimiento por iteración.

**Figura 4.** Mejor solución global en mapa real de México.

![Figura 4 - Mejor solución global en mapa real](opt.%20combinatoria/outputs/mejor_ruta_global.png)

La Figura 4 sintetiza la ruta final de menor costo observada, y complementa la conclusión cuantitativa reportada en la Tabla 5.

### 2.6 Recomendación al vendedor viajero (orden específico)
**Recomendación principal (global):**
Usar la ruta de ACO para `valor_hora=100`.

Orden recomendado de visita:
Ciudad de México -> Toluca -> Cuernavaca -> Puebla -> Tlaxcala -> Xalapa -> Villahermosa -> San Francisco de Campeche -> Mérida -> Chetumal -> Tuxtla Gutiérrez -> Oaxaca de Juárez -> Chilpancingo -> Morelia -> Santiago de Querétaro -> Guanajuato -> San Luis Potosí -> Aguascalientes -> Zacatecas -> Guadalajara -> Colima -> Tepic -> Durango -> Culiacán -> La Paz -> Hermosillo -> Mexicali -> Chihuahua -> Saltillo -> Monterrey -> Ciudad Victoria -> Pachuca -> Ciudad de México.

**Escenario alterno (cuando se prioriza un patrón GA competitivo):**
Para `valor_hora=250`, GA logra mejor mejor-caso que ACO (74,821.81 vs 75,035.99 MXN). Orden GA en ese escenario:
Ciudad de México -> Xalapa -> Villahermosa -> San Francisco de Campeche -> Mérida -> Chetumal -> Tuxtla Gutiérrez -> Oaxaca de Juárez -> Chilpancingo -> Toluca -> Cuernavaca -> Puebla -> Tlaxcala -> Pachuca -> Santiago de Querétaro -> Morelia -> Colima -> Guadalajara -> Tepic -> Durango -> Culiacán -> La Paz -> Hermosillo -> Mexicali -> Chihuahua -> Saltillo -> Monterrey -> Ciudad Victoria -> San Luis Potosí -> Zacatecas -> Aguascalientes -> Guanajuato -> Ciudad de México.

## 3) Metodología y justificación técnica
- Se usaron funciones benchmark clásicas para comparar métodos con distintas geometrías del paisaje de optimización (Rastrigin y Rosenbrock).
- Se estandarizó el número de iteraciones y se evaluó robustez mediante múltiples corridas con diferentes semillas.
- En TSP se modeló costo monetario total por arco (tiempo, peajes y combustible) con snapshot trazable (`costos_matriz.csv`) y metadatos de fuente (`costos_fuentes.json`), además de sensibilidad al valor-hora.
- Se priorizó reproducibilidad mediante configuración explícita y artefactos exportados (CSV/GIF/PNG).

## 4) Uso de IA
Registrar prompts principales y su impacto real en el resultado.

| ID | Prompt usado | Objetivo | Resultado obtenido | Impacto en calidad/final |
|---|---|---|---|---|
| P1 | Genera una visualización animada (GIF) del mejor recorrido TSP sobre un mapa de México: dibuja las capitales con lat/lon, traza la ruta iteración a iteración y guarda mejor_ruta_global.gif y mejor_ruta_global.png con anotaciones de costo, seed y algoritmo. | Generar gifs para acelerar tiempo de desarrollo y usar razonamiento en objetivos mas importantes. | Gifs representando el mejor recorrido en el mapa de México. | Medio |
| P2 | Propón hiperparámetros iniciales para ACO (alpha, beta, evaporación, q, número de hormigas) orientados a TSP de 32 nodos. | Definir una configuración inicial razonable de ACO para un TSP de 32 ciudades, que balancee exploración y explotación. | Se usó num_ants=55, iterations=120, alpha=1.0, beta=3.0, evaporation=0.35, q=120.0 (config actual), con desempeño competitivo en casi todo el barrido de valor_hora, incluyendo el mejor costo global del experimento (46,698.84 MXN a valor_hora=100) | medio |

Guía de análisis:
- Qué tareas aceleró la IA.
- Qué errores/sesgos introdujo.
- Cómo se validó o corrigió la salida.

## 5) Conclusiones
- En Parte 1, los métodos heurísticos (en especial DE y PSO) alcanzaron mejores mínimos promedio que GD en los escenarios evaluados.
- GD tuvo ventaja en costo de evaluación, pero menor calidad de solución en promedio para los casos más difíciles.
- En Parte 2, ACO obtuvo la mejor solución global (`valor_hora=100`), y GA solo superó el mejor-caso de ACO en el escenario `valor_hora=250`, con mayor variabilidad entre semillas.
- La modelación del costo con `valor_hora` cambió de forma significativa la ruta/costo óptimos, por lo que este parámetro es clave en análisis de sensibilidad.

## 7) Bibliografia

1. Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. The Computer Journal, 3(3), 175-184. https://doi.org/10.1093/comjnl/3.3.175
2. Nocedal, J., & Wright, S. J. (2006). Numerical optimization (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5
3. Boyd, S., & Vandenberghe, L. (2004). Convex optimization. Cambridge University Press. https://stanford.edu/~boyd/cvxbook/
4. Holland, J. H. (1992). Adaptation in natural and artificial systems: An introductory analysis with applications to biology, control, and artificial intelligence. MIT Press. https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/
5. Mitchell, M. (1998). An introduction to genetic algorithms. MIT Press. https://mitpress.mit.edu/9780262631853/an-introduction-to-genetic-algorithms/
6. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. In Proceedings of ICNN'95 - International Conference on Neural Networks (Vol. 4, pp. 1942-1948). IEEE. https://doi.org/10.1109/ICNN.1995.488968
7. Storn, R., & Price, K. (1997). Differential evolution-a simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization, 11, 341-359. https://doi.org/10.1023/A:1008202821328
8. Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: Optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 26(1), 29-41. https://doi.org/10.1109/3477.484436
9. Jamil, M., & Yang, X.-S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194. https://doi.org/10.1504/IJMMNO.2013.055204
10. Surjanovic, S., & Bingham, D. (2013). Virtual library of simulation experiments: Test functions and datasets. Simon Fraser University. https://www.sfu.ca/~ssurjano/optimization.html
11. Surjanovic, S., & Bingham, D. (2013). Rosenbrock function. Simon Fraser University. https://www.sfu.ca/~ssurjano/rosen.html
12. Surjanovic, S., & Bingham, D. (2013). Rastrigin function. Simon Fraser University. https://www.sfu.ca/~ssurjano/rastr.html
13. Surjanovic, S., & Bingham, D. (2013). Schwefel function. Simon Fraser University. https://www.sfu.ca/~ssurjano/schwef.html
14. Surjanovic, S., & Bingham, D. (2013). Griewank function. Simon Fraser University. https://www.sfu.ca/~ssurjano/griewank.html
15. Surjanovic, S., & Bingham, D. (2013). Goldstein-Price function. Simon Fraser University. https://www.sfu.ca/~ssurjano/goldpr.html
16. Surjanovic, S., & Bingham, D. (2013). Six-Hump Camel function. Simon Fraser University. https://www.sfu.ca/~ssurjano/camel6.html
17. Applegate, D. L., Bixby, R. E., Chvatal, V., & Cook, W. J. (2007). The traveling salesman problem: A computational study. Princeton University Press. https://press.princeton.edu/books/hardcover/9780691129938/the-traveling-salesman-problem
18. Instituto Nacional de Estadistica y Geografia (INEGI). (2026). Servicio web del Catalogo Unico de Claves Geoestadisticas. https://www.inegi.org.mx/servicios/catalogounico.html
19. Secretaria de Infraestructura, Comunicaciones y Transportes (SICT). (2011). Carreteras V2 (incluye modulo Traza tu ruta). https://www.sct.gob.mx/index.php?id=1617
20. Caminos y Puentes Federales de Ingresos y Servicios Conexos (CAPUFE). (2025). Tarifas CAPUFE [Conjunto de datos]. datos.gob.mx. https://www.datos.gob.mx/dataset/tarifas_capufe
21. Comision Nacional de Energia (CNE). (2025). Historico de precios de gasolinas y diesel reportados por permisionario [Conjunto de datos]. datos.gob.mx. https://www.datos.gob.mx/dataset/historico_precios_gasolinas_y_diesel_reportados_por_permisionario
22. Instituto Nacional de Estadistica y Geografia (INEGI). (2026). API de Ruteo (SAKBE v3.1). https://www.inegi.org.mx/servicios/Ruteo/










