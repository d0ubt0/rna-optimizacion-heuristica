# RNA - Optimizacion Heuristica

Proyecto academico del curso **Redes Neuronales y Algoritmos Bioinspirados** (UNAL), centrado en comparar metodos de optimizacion continua y discreta bajo criterios de calidad de solucion, estabilidad entre corridas y costo computacional.

## Integrantes

- Jhofred Jahat Camacho Gomez
- Sebastian Pabon Nunez

## Descripcion del problema

El trabajo se divide en dos partes:

1. **Parte 1 - Optimizacion numerica**
- Seleccion de 2 funciones benchmark de una lista clasica.
- Optimizacion en 2D y 3D con:
  - Descenso por gradiente (inicio aleatorio).
  - Algoritmo evolutivo (EA).
  - Particle Swarm Optimization (PSO).
  - Evolucion diferencial (DE).
- Generacion de animaciones (GIF/video) del proceso de convergencia.
- Comparacion por valor final de la funcion objetivo y numero de evaluaciones.

2. **Parte 2 - Optimizacion combinatoria (TSP Mexico)**
- Vendedor que debe visitar las 32 capitales estatales de Mexico.
- Resolucion con ACO y GA.
- Modelo de costo por tramo:
  - costo por tiempo (valor hora),
  - peajes,
  - combustible (segun vehiculo seleccionado).
- Analisis de sensibilidad del valor-hora.
- Visualizacion de la mejor ruta sobre mapa real de Mexico (GIF/video).

## Alcance implementado en este repositorio

### Parte 1 (optimizacion numerica)

Este repositorio implementa la comparacion sobre **Rastrigin** y **Rosenbrock** en 2D y 3D, con 20 corridas por caso (`opt. numerica/resultados_parte1_20corridas.csv`).

Componentes principales:

- `opt. numerica/experimento_parte1.py`: ejecuta el benchmark completo y exporta metricas agregadas.
- `opt. numerica/gradiente.py`: descenso por gradiente.
- `opt. numerica/evolutivo.py`: algoritmo evolutivo.
- `opt. numerica/particulas.py`: PSO.
- `opt. numerica/evolucion_diferencial.py`: DE.
- `opt. numerica/funciones.py`: funciones objetivo, gradientes y utilidades de visualizacion/animacion.

Metricas registradas por caso:

- promedio del valor final,
- desviacion estandar,
- mejor y peor corrida,
- evaluaciones promedio de la funcion objetivo.

Lectura general de resultados (20 corridas):

- En **Rastrigin 2D/3D**, los metodos heuristicas (DE/EA/PSO) superan claramente a GD en calidad final.
- En **Rosenbrock 2D**, DE y PSO alcanzan valores muy cercanos al optimo global.
- En **Rosenbrock 3D**, DE y PSO mantienen mejor robustez que EA y GD.
- GD usa menos evaluaciones por corrida (101), pero con mayor riesgo de quedar en soluciones locales en paisajes multimodales.

### Parte 2 (optimizacion combinatoria)

La implementacion de TSP para Mexico usa matrices de costos y configuracion parametrica para comparar ACO y GA con multiples semillas y valores-hora.

Componentes principales:

- `opt. combinatoria/src/main.py`: punto de entrada del experimento + visualizaciones.
- `opt. combinatoria/src/experiment.py`: orquestacion de corridas ACO/GA y consolidacion de resultados.
- `opt. combinatoria/src/cost_model.py`: construccion de matriz de costo total.
- `opt. combinatoria/src/aco_solver.py`: solucionador ACO.
- `opt. combinatoria/src/ga_solver.py`: solucionador GA.
- `opt. combinatoria/src/visualize.py`: graficos y GIF de rutas sobre mapa de Mexico.
- `opt. combinatoria/data/config.yaml`: parametros del experimento.

Configuracion actual destacada:

- capital inicial: Ciudad de Mexico,
- vehiculo: `sedan_gasolina`,
- barrido de valor-hora: 100 a 300 MXN/h (paso 50),
- semillas: 7, 13, 29,
- salida de metricas comparativas y mejores rutas por algoritmo/valor-hora.

Artefactos generados:

- `opt. combinatoria/outputs/comparativa_metricas.csv`
- `opt. combinatoria/outputs/mejor_ruta_aco*.csv`
- `opt. combinatoria/outputs/mejor_ruta_ga*.csv`
- `opt. combinatoria/outputs/ruta_ga_iteraciones_mapa_real.gif`
- `opt. combinatoria/outputs/mejor_ruta_global.gif`

## Estructura del repositorio

- `opt. numerica/`: codigo y resultados de optimizacion continua.
- `opt. combinatoria/`: codigo, datos y salidas del problema TSP.
- `reporte_tecnico_blog.md`: reporte tecnico consolidado.
- `bibliografia_apa.txt`: referencias bibliograficas.
- `requirements.txt`: dependencias Python.

## Ejecucion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar Parte 1 (benchmark numerico):

```bash
python "opt. numerica/experimento_parte1.py"
```

Ejecutar Parte 2 (experimento combinatorio + visualizaciones):

```bash
cd "opt. combinatoria"
python -m src.main --config data/config.yaml --data-dir data --output-dir outputs
```

## Discusion tecnica (resumen)

- **Descenso por gradiente** aporta eficiencia en evaluaciones y simplicidad cuando el paisaje es suave y bien condicionado.
- **Heuristicas poblacionales** (EA/PSO/DE/ACO/GA) aportan mayor exploracion global y mejor desempeno en problemas multimodales o discretos complejos.
- El trade-off principal del proyecto es **costo computacional vs. calidad/robustez de la solucion**.
