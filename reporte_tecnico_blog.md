# Reporte técnico (entrada de blog): Optimización numérica y combinatoria

**Curso:** Redes Neuronales y Algoritmos Bioinspirados
**Equipo:** Sebastián Pabón Núñez, Jhofred Jahat Camacho Gómez
**Fecha:** 24/03/2026
**Repositorio de código:** [Github](https://github.com/d0ubt0/rna-optimizacion-heuristica)

## Resumen ejecutivo
Se resolvieron dos problemas: (1) optimización numérica en funciones de prueba continuas y (2) optimización combinatoria de rutas en las 32 capitales estatales de México. Para la Parte 1 se compararon descenso por gradiente, algoritmo evolutivo (EA), PSO y evolución diferencial (DE), en 2D y 3D, con 20 corridas por caso. Para la Parte 2 se compararon ACO y GA variando el valor-hora del vendedor, con costos de tiempo, peajes y combustible.

## 1) Parte 1: Optimización numérica


### 1. Introducción
El presente reporte detalla el estudio y comparación de diversos algoritmos de optimización aplicados a dos funciones de prueba clásicas en el ámbito del análisis numérico: **Rosenbrock** (conocida por su estrecho valle) y **Rastrigin** (caracterizada por su alta multimodalidad). El objetivo es evaluar el desempeño de métodos basados en gradiente frente a aproximaciones heurísticas en espacios de búsqueda de 2 y 3 dimensiones.

---

### 2. Metodología
Para resolver los problemas de optimización, se implementaron cuatro enfoques distintos:

* **Descenso por Gradiente (GD):** Un método iterativo de optimización de primer orden que se desplaza en la dirección opuesta al gradiente de la función en el punto actual. Se utilizó una condición inicial aleatoria.
* **Algoritmos Evolutivos (EA):** Métodos de búsqueda estocástica inspirados en la evolución biológica (selección, mutación y recombinación) para explorar el espacio de soluciones mediante una población.
* **Optimización por Enjambre de Partículas (PSO):** Algoritmo basado en el comportamiento social de bandadas de aves o bancos de peces, donde "partículas" se mueven en el espacio ajustando su posición según su mejor experiencia individual y la del grupo.
* **Evolución Diferencial (DE):** Un tipo de algoritmo evolutivo que optimiza problemas mediante la combinación de vectores de diferencia de la población actual para crear nuevos candidatos.

---

### 3. Resultados
A continuación, se presentan las métricas obtenidas tras múltiples ejecuciones de cada experimento:

| Experimento | Promedio | Mediana | Std | Mínimo | Máximo | Tasa Éxito | Tiempo Medio (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GD Rastrigin 2D | 17.5111 | 18.9041 | 8.5380 | 3.9798 | 28.8535 | 0.0000 | 0.0006 |
| GD Rastrigin 3D | 27.3611 | 27.8586 | 12.2604 | 4.9747 | 49.7474 | 0.0000 | 0.0005 |
| GD Rosenbrock 2D | 1.0698 | 0.6317 | 1.3993 | 0.0041 | 4.8054 | 0.1000 | 0.0005 |
| GD Rosenbrock 3D | 1.3080 | 1.3189 | 1.1161 | 0.0057 | 3.9595 | 0.1000 | 0.0007 |
| DE Rastrigin 2D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0934 |
| DE Rastrigin 3D | 1.2715 | 1.1292 | 0.7849 | 0.2713 | 3.2107 | 0.0000 | 0.0938 |
| DE Rosenbrock 2D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0969 |
| DE Rosenbrock 3D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0970 |
| EA Rastrigin 2D | 0.0001 | 0.0000 | 0.0003 | 0.0000 | 0.0011 | 1.0000 | 0.2348 |
| EA Rastrigin 3D | 0.0003 | 0.0001 | 0.0007 | 0.0000 | 0.0025 | 1.0000 | 0.2371 |
| EA Rosenbrock 2D | 0.0558 | 0.0021 | 0.0771 | 0.0000 | 0.2048 | 0.6000 | 0.2394 |
| EA Rosenbrock 3D | 0.4883 | 0.5422 | 0.3390 | 0.0019 | 0.9297 | 0.2000 | 0.2426 |
| PSO Rastrigin 2D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0268 |
| PSO Rastrigin 3D | 0.3010 | 0.0008 | 0.4542 | 0.0000 | 0.9949 | 0.6000 | 0.0270 |
| PSO Rosenbrock 2D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0307 |
| PSO Rosenbrock 3D | 0.0585 | 0.0697 | 0.0422 | 0.0000 | 0.1179 | 0.3000 | 0.0308 |

---

### 4. Visualización del proceso

#### Función Rastrigin (2D y 3D)
Representación de la convergencia en una superficie altamente multimodal.

* **Descenso por Gradiente:** Se observa cómo queda atrapado en óptimos locales fácilmente.
    ![Rastrigin GD](opt.%20numerica/animaciones/animacion_rastrigin.gif)
    ![Rastrigin 3D GD](opt.%20numerica/animaciones/animacion_3d_rastrigin_3d.gif)

* **Evolución Diferencial (DE):**
    ![Rastrigin DE](opt.%20numerica/animaciones/animacion_rastrigin_de.gif)
    ![Rastrigin 3D DE](opt.%20numerica/animaciones/animacion_3d_rastrigin_3d_de.gif)

* **Algoritmos Evolutivos (EA):**
    ![Rastrigin EA](opt.%20numerica/animaciones/animacion_rastrigin_ea.gif)
    ![Rastrigin 3D EA](opt.%20numerica/animaciones/animacion_3d_rastrigin_3d_ea.gif)

* **PSO:**
    ![Rastrigin PSO](opt.%20numerica/animaciones/animacion_rastrigin_pso.gif)
    ![Rastrigin 3D PSO](opt.%20numerica/animaciones/animacion_3d_rastrigin_3d_pso.gif)

#### Función Rosenbrock (2D y 3D)
Visualización de la búsqueda en el valle parabólico.

* **Descenso por Gradiente:**
    ![Rosenbrock GD](opt.%20numerica/animaciones/animacion_rosenbrock.gif)
    ![Rosenbrock 3D GD](opt.%20numerica/animaciones/animacion_3d_rosenbrock_3d.gif)

* **Evolución Diferencial (DE):**
    ![Rosenbrock DE](opt.%20numerica/animaciones/animacion_rosenbrock_de.gif)
    ![Rosenbrock 3D DE](opt.%20numerica/animaciones/animacion_3d_rosenbrock_3d_de.gif)

* **Algoritmos Evolutivos (EA):**
    ![Rosenbrock EA](opt.%20numerica/animaciones/animacion_rosenbrock_ea.gif)
    ![Rosenbrock 3D EA](opt.%20numerica/animaciones/animacion_3d_rosenbrock_3d_ea.gif)

* **PSO:**
    ![Rosenbrock PSO](opt.%20numerica/animaciones/animacion_rosenbrock_pso.gif)
    ![Rosenbrock 3D PSO](opt.%20numerica/animaciones/animacion_3d_rosenbrock_3d_pso.gif)

---

### 5. Discusión

#### Análisis de Métodos
1.  **¿Qué aportan los métodos de gradiente?**
    Aportan una velocidad computacional extrema. Como se observa en la columna `Tiempo Medio (s)`, el Descenso por Gradiente es órdenes de magnitud más rápido (~0.0006s) que los heurísticos. Sin embargo, su tasa de éxito es nula en Rastrigin, ya que al no tener visión global, convergen al mínimo local más cercano a su punto de inicio aleatorio.

2.  **¿Qué aportan los heurísticos?**
    Aportan robustez y capacidad de exploración global. A pesar de requerir un mayor número de evaluaciones (reflejado en tiempos de ejecución superiores, entre 0.02s y 0.24s), logran tasas de éxito de 1.0 (100%) en problemas donde el gradiente falla.

#### Comparativa y Estabilidad
* **Convergencia:** DE y EA muestran una convergencia muy estable hacia el valor óptimo (promedios cercanos a 0 en Rastrigin y 1 en Rosenbrock).
* **Estabilidad:** El PSO destaca por un excelente equilibrio entre velocidad (siendo el más rápido de los heurísticos con ~0.02s) y precisión, aunque su tasa de éxito disminuye en dimensiones superiores (3D).

> **Resumen Ventajas/Desventajas:**
> * **GD:** Ventaja: Rapidez. Desventaja: Inestabilidad en funciones no convexas.
> * **DE/EA:** Ventaja: Alta probabilidad de encontrar el óptimo global. Desventaja: Alto costo computacional.

---

### 6. Conclusiones
La elección del algoritmo depende de la naturaleza de la función. Para funciones simples o convexas, el gradiente es imbatible por su eficiencia. No obstante, para problemas complejos y multimodales como Rastrigin, los métodos heurísticos (especialmente la Evolución Diferencial y el PSO) son indispensables para garantizar el hallazgo de la solución óptima, sacrificando tiempo de cómputo por fiabilidad.
## 2) Optimización combinatoria 

### 1. Definición del problema
Un vendedor debe visitar todas las capitales de los 32 estados de México y regresar al origen (Ciudad de México en esta configuración).

---

### 2. Modelado de costo
El costo entre ciudades se modeló como:
$$C_{ij} = (valor\_hora \cdot tiempo_{ij}) + peajes_{ij} + combustible_{ij}$$

donde:

$$combustible_{ij} = distancia_{ij} \cdot \frac{precio\_litro}{rendimiento_{km/L}}$$

---

### 3. Vehículo y parámetro estudiado
- Vehículo seleccionado: **Sedan Gasolina** (`vehicle_id=sedan_gasolina`).
- Rendimiento: **15.5 km/L**.
- Precio combustible: **24.2 MXN/L**.
- Parámetro analizado: `valor_hora` en **[100, 300] MXN/h** con paso **50**.

---

### 4. Métodos implementados
- Colonia de hormigas (ACO).
- Algoritmo genético (GA).

Configuración usada (archivo `opt. combinatoria/data/config.yaml`):
- Seeds: `[7, 13, 29]`
- ACO: `num_ants=55`, `iterations=120`, `alpha=1.0`, `beta=3.0`, `evaporation=0.35`, `q=120.0`.
- GA: `population_size=140`, `generations=220`, `crossover_rate=0.9`, `mutation_rate=0.22`, `elite_size=4`, `tournament_size=4`.


---


### 5. Resultados y visualización
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
| P1 | Genera una visualización animada (GIF) del mejor recorrido TSP sobre un mapa de México: dibuja las capitales con lat/lon, traza la ruta iteración a iteración y guarda mejor_ruta_global.gif y mejor_ruta_global.png con anotaciones de costo, seed y algoritmo. | Generar gifs para acelerar tiempo de desarrollo y usar razonamiento en objetivos mas importantes. | Gifs representando el mejor recorrido en el mapa de México. | Medio |
| P2 | Propón hiperparámetros iniciales para ACO (alpha, beta, evaporación, q, número de hormigas) orientados a TSP de 32 nodos. | Definir una configuración inicial razonable de ACO para un TSP de 32 ciudades, que balancee exploración y explotación. | Se usó num_ants=55, iterations=120, alpha=1.0, beta=3.0, evaporation=0.35, q=120.0 (config actual), con desempeño competitivo en casi todo el barrido de valor_hora, incluyendo el mejor costo global del experimento (48,133.25 MXN a valor_hora=100) | medio |

## 5) Video de contribución individual (obligatorio)
Incluir URL del video final y aportes en primera persona.

- URL video: https://drive.google.com/file/d/1_WWLZ-UhydpCneBUtrUm_3TcAOnLVVLS/view?usp=sharing

## 6) Conclusiones
- En Parte 1, los métodos heurísticos (en especial DE y PSO) alcanzaron mejores mínimos promedio que GD en los escenarios evaluados.
- GD tuvo ventaja en costo de evaluación, pero menor calidad de solución en promedio para los casos más difíciles.
- En Parte 2, ACO fue globalmente más competitivo para `valor_hora` bajos e intermedios; GA mostró mejor mejor-caso en `valor_hora=300`, con mayor variabilidad.
- La modelación del costo con `valor_hora` cambió de forma significativa la ruta/costo óptimos, por lo que este parámetro es clave en análisis de sensibilidad.

## 7) Bibliograf?a

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
