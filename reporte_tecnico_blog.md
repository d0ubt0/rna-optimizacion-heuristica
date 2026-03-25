# Reporte técnico (entrada de blog): Optimización numérica y combinatoria

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Equipo:** Sebastián Pabón Núñez, Jhofred Jahat Camacho Gómez  
**Fecha:** 24/03/2026  
**Repositorio de código:** [Github](https://github.com/d0ubt0/rna-optimizacion-heuristica)

> Este documento sigue los requisitos del enunciado.  
> La bibliografía en formato APA con enlaces reales está en `bibliografia_apa.txt`.

## Resumen ejecutivo
Se resolvieron dos problemas: (1) optimización numérica en funciones de prueba continuas y (2) optimización combinatoria de rutas en las 32 capitales estatales de México. Para la Parte 1 se compararon descenso por gradiente, algoritmo evolutivo (EA), PSO y evolución diferencial (DE), en 2D y 3D, con 20 corridas por caso. Para la Parte 2 se compararon ACO y GA variando el valor-hora del vendedor, con costos de tiempo, peajes y combustible.

## 1) Parte 1: Optimización numérica

### 1.1 Funciones seleccionadas
De la lista propuesta se eligieron:
- **Función 1:** Rastrigin.
- **Función 2:** Rosenbrock.

**Justificación de selección:**
- Rastrigin es altamente multimodal y estresa la exploración global.
- Rosenbrock tiene un valle estrecho y curvo, útil para evaluar estabilidad y precisión de convergencia.

### 1.2 Métodos implementados
Se resolvió cada función en 2D y 3D con:
1. Descenso por gradiente (GD) con condición inicial aleatoria.
2. Algoritmo evolutivo (EA).
3. Optimización por enjambre de partículas (PSO).
4. Evolución diferencial (DE).

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

#### Tabla 1. Rastrigin 2D (20 corridas)
| Método | Promedio f(x) | Desviación | Mejor | Evaluaciones |
|---|---:|---:|---:|---:|
| GD | 18.108137 | 9.982277 | 1.989918 | 101 |
| EA | 4.350e-05 | 8.693e-05 | 7.105e-15 | 5001 |
| PSO | 5.885e-07 | 2.517e-06 | 2.165e-11 | 3031 |
| DE | 0.099496 | 0.298488 | 7.105e-15 | 2021 |

#### Tabla 2. Rastrigin 3D (20 corridas)
| Método | Promedio f(x) | Desviación | Mejor | Evaluaciones |
|---|---:|---:|---:|---:|
| GD | 26.067764 | 12.240703 | 4.974790 | 101 |
| EA | 7.814e-04 | 0.002294 | 7.015e-08 | 5001 |
| PSO | 0.151540 | 0.354368 | 2.003e-09 | 3031 |
| DE | 0.717205 | 0.600599 | 4.775e-05 | 2021 |

#### Tabla 3. Rosenbrock 2D (20 corridas)
| Método | Promedio f(x) | Desviación | Mejor | Evaluaciones |
|---|---:|---:|---:|---:|
| GD | 1.294852 | 1.506695 | 0.001980 | 101 |
| EA | 0.061635 | 0.112428 | 2.932e-06 | 5001 |
| PSO | 5.848e-06 | 1.310e-05 | 4.531e-10 | 3031 |
| DE | 5.507e-15 | 1.019e-14 | 8.828e-19 | 2021 |

#### Tabla 4. Rosenbrock 3D (20 corridas)
| Método | Promedio f(x) | Desviación | Mejor | Evaluaciones |
|---|---:|---:|---:|---:|
| GD | 1.408352 | 1.021637 | 0.005781 | 101 |
| EA | 0.737961 | 0.572848 | 0.184525 | 5001 |
| PSO | 0.089464 | 0.047688 | 4.998e-05 | 3031 |
| DE | 0.002521 | 0.010925 | 8.549e-09 | 2021 |

Fuente de datos de tablas: `opt. numerica/resultados_parte1_20corridas.csv`.

### 1.5 Animaciones (obligatorio)
- **GIF gradiente:** [animacion_gd_rastrigin_2d.gif](opt.%20numerica/outputs/animacion_gd_rastrigin_2d.gif)
- **GIF heur?stico (DE):** [animacion_de_rastrigin_2d.gif](opt.%20numerica/outputs/animacion_de_rastrigin_2d.gif)

**Figura 1.** Trayectoria de optimizaci?n de descenso por gradiente en Rastrigin 2D.

![Figura 1 - GD Rastrigin 2D](opt.%20numerica/outputs/animacion_gd_rastrigin_2d.gif)

**Figura 2.** Trayectoria de optimizaci?n por evoluci?n diferencial en Rastrigin 2D.

![Figura 2 - DE Rastrigin 2D](opt.%20numerica/outputs/animacion_de_rastrigin_2d.gif)

### 1.6 Discusión solicitada
- **Aporte de GD:** menor costo computacional por corrida (101 evaluaciones), implementación directa cuando existe gradiente analítico, y convergencia razonable en regiones suaves.
- **Aporte de heurísticos:** mejores mínimos finales en problemas multimodales y no convexos; en particular DE/PSO/EA superan ampliamente a GD en Rastrigin y Rosenbrock 3D.
- **Trade-off principal:** GD usa muchas menos evaluaciones, pero obtiene peores valores finales promedio en la mayoría de los casos de este estudio.
- **Necesidad de múltiples corridas:** sí. La variabilidad observada (desviaciones no nulas) confirma sensibilidad a semilla e inicialización en métodos heurísticos y también en GD por inicialización aleatoria.

## 2) Parte 2: Optimización combinatoria (TSP de 32 capitales)

### 2.1 Definición del problema
Un vendedor debe visitar todas las capitales de los 32 estados de México y regresar al origen (Ciudad de México en esta configuración).

### 2.2 Modelado de costo
El costo entre ciudades se modeló como:

\[
C_{ij} = (valor\_hora \cdot tiempo_{ij}) + peajes_{ij} + combustible_{ij}
\]

con

\[
combustible_{ij} = distancia_{ij} \cdot \frac{precio\_litro}{rendimiento\_{km/L}}
\]

### 2.3 Vehículo y parámetro estudiado
- Vehículo seleccionado: **Sedan Gasolina** (`vehicle_id=sedan_gasolina`).
- Rendimiento: **15.5 km/L**.
- Precio combustible: **24.2 MXN/L**.
- Parámetro analizado: `valor_hora` en **[100, 300] MXN/h** con paso **50**.

### 2.4 Métodos implementados
- Colonia de hormigas (ACO).
- Algoritmo genético (GA).

Configuración usada (archivo `opt. combinatoria/data/config.yaml`):
- Seeds: `[7, 13, 29]`
- ACO: `num_ants=55`, `iterations=120`, `alpha=1.0`, `beta=3.0`, `evaporation=0.35`, `q=120.0`.
- GA: `population_size=140`, `generations=220`, `crossover_rate=0.9`, `mutation_rate=0.22`, `elite_size=4`, `tournament_size=4`.

### 2.5 Resultados y visualización
#### Tabla 5. Comparativa ACO vs GA por valor-hora
| Valor hora (MXN/h) | Algoritmo | Mejor costo (MXN) | Promedio (MXN) | Desviación | Seed mejor |
|---:|---|---:|---:|---:|---:|
| 100 | ACO | 48133.25 | 48236.41 | 126.46 | 29 |
| 100 | GA | 51941.51 | 52029.39 | 112.16 | 29 |
| 150 | ACO | 57419.85 | 57743.30 | 229.02 | 13 |
| 150 | GA | 61384.18 | 62505.10 | 823.67 | 13 |
| 200 | ACO | 67191.28 | 67303.23 | 79.36 | 7 |
| 200 | GA | 69230.55 | 71016.66 | 1813.90 | 7 |
| 250 | ACO | 75694.35 | 76722.94 | 801.81 | 13 |
| 250 | GA | 78371.23 | 81912.59 | 2577.75 | 13 |
| 300 | ACO | 85506.73 | 85830.92 | 261.38 | 7 |
| 300 | GA | 85065.01 | 89875.79 | 3437.38 | 7 |

**Mejor solución global observada:** ACO con `valor_hora=100`, costo `48,133.25 MXN`.

Artefactos:
- **GIF de la mejor ruta:** [mejor_ruta_global.gif](opt.%20combinatoria/outputs/mejor_ruta_global.gif)
- **Figura final de ruta:** [mejor_ruta_global.png](opt.%20combinatoria/outputs/mejor_ruta_global.png)
- **Rutas por m?todo:** [mejor_ruta_aco.csv](opt.%20combinatoria/outputs/mejor_ruta_aco.csv), [mejor_ruta_ga.csv](opt.%20combinatoria/outputs/mejor_ruta_ga.csv)

**Figura 3.** Recorrido final sobre mapa de M?xico.

![Figura 3 - Recorrido final en mapa de M?xico](opt.%20combinatoria/outputs/mejor_ruta_global.png)

**Figura 4.** Evoluci?n iterativa de la mejor soluci?n (GIF).

![Figura 4 - Evoluci?n de la mejor soluci?n TSP](opt.%20combinatoria/outputs/mejor_ruta_global.gif)

## 3) Metodología y justificación técnica
- Se usaron funciones benchmark clásicas para comparar métodos con distintas geometrías del paisaje de optimización (Rastrigin y Rosenbrock).
- Se estandarizó el número de iteraciones y se evaluó robustez mediante múltiples corridas con diferentes semillas.
- En TSP se modeló costo monetario total por arco incorporando tiempo, peajes y combustible, y se estudió sensibilidad al valor-hora.
- Se priorizó reproducibilidad mediante configuración explícita y artefactos exportados (CSV/GIF/PNG).

## 4) Uso de IA (obligatorio, plantilla editable)
Registrar prompts principales y su impacto real en el resultado.

| ID | Prompt usado | Objetivo | Resultado obtenido | Impacto en calidad/final |
|---|---|---|---|---|
| P1 | [completar] | [completar] | [completar] | [alto/medio/bajo + explicación] |
| P2 | [completar] | [completar] | [completar] | [alto/medio/bajo + explicación] |
| P3 | [completar] | [completar] | [completar] | [alto/medio/bajo + explicación] |

Guía de análisis:
- Qué tareas aceleró la IA.
- Qué errores/sesgos introdujo.
- Cómo se validó o corrigió la salida.

## 5) Video de contribución individual (obligatorio)
Incluir URL del video final y aportes en primera persona.

- Sebastián Pabón Núñez: [completar aporte específico en primera persona].
- Jhofred Jahat Camacho Gómez: [completar aporte específico en primera persona].
- URL video: [completar]

## 6) Publicación como blog
Este documento está estructurado para publicación en GitHub (README o GitHub Pages):
1. Contexto y objetivo.
2. Metodología.
3. Resultados (tablas y GIF).
4. Discusión.
5. Conclusiones.
6. Repositorio y bibliografía.

## 7) Conclusiones
- En Parte 1, los métodos heurísticos (en especial DE y PSO) alcanzaron mejores mínimos promedio que GD en los escenarios evaluados.
- GD tuvo ventaja en costo de evaluación, pero menor calidad de solución en promedio para los casos más difíciles.
- En Parte 2, ACO fue globalmente más competitivo para `valor_hora` bajos e intermedios; GA mostró mejor mejor-caso en `valor_hora=300`, con mayor variabilidad.
- La modelación del costo con `valor_hora` cambió de forma significativa la ruta/costo óptimos, por lo que este parámetro es clave en análisis de sensibilidad.

## 8) Bibliograf?a

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

## Checklist de cumplimiento
- [x] Dos funciones de prueba elegidas y justificadas.
- [x] Optimización en 2D y 3D con gradiente.
- [x] Optimización en 2D y 3D con evolutivo, PSO y DE.
- [x] GIF/video de gradiente y heurístico (Parte 1).
- [x] ACO y GA para TSP de 32 capitales.
- [x] Estudio del parámetro valor-hora.
- [x] Vehículo definido y costo combustible documentado.
- [x] GIF/video de la mejor ruta en mapa de México (Parte 2).
- [x] Bibliografía en APA con enlaces reales.
- [ ] Prompts de IA reportados y discutidos (pendiente de completar por el equipo).
- [ ] Video de contribución individual incluido (pendiente de enlace final).
- [x] Repositorio Git referenciado.
