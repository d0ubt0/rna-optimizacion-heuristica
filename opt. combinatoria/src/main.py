import argparse
from pathlib import Path

from .experiment import run_experiment
from .visualize import create_route_animation


def main() -> None:
    parser = argparse.ArgumentParser(description='TSP Mexico with ACO and GA')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('opt. combinatoria/data/config.yaml'),
        help='Path to config file (JSON content accepted in .yaml)',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('opt. combinatoria/data'),
        help='Path to data directory',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('opt. combinatoria/outputs'),
        help='Path to output directory',
    )

    args = parser.parse_args()

    result_bundle = run_experiment(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    config = result_bundle['config']
    viz_cfg = config.get('visualization', {})
    geojson_name = str(viz_cfg.get('mexico_geojson', 'mexico_border.geojson'))
    mexico_geojson_path = Path(geojson_name)
    if not mexico_geojson_path.is_absolute():
        mexico_geojson_path = args.data_dir / mexico_geojson_path

    create_route_animation(
        capitals=result_bundle['capitals'],
        result=result_bundle['best_ga'],
        output_gif=args.output_dir / 'ruta_ga_iteraciones_mapa_real.gif',
        output_png=args.output_dir / 'ruta_ga_iteraciones_mapa_real.png',
        mexico_geojson_path=mexico_geojson_path,
        fps=int(viz_cfg.get('fps', 6)),
        figsize=viz_cfg.get('figsize', [10, 8]),
    )

    create_route_animation(
        capitals=result_bundle['capitals'],
        result=result_bundle['best_global'],
        output_gif=args.output_dir / 'mejor_ruta_global.gif',
        output_png=args.output_dir / 'mejor_ruta_global.png',
        mexico_geojson_path=mexico_geojson_path,
        fps=int(viz_cfg.get('fps', 6)),
        figsize=viz_cfg.get('figsize', [10, 8]),
    )

    best = result_bundle['best_global']
    print('==== RESUMEN ====')
    print(f"Vehiculo: {result_bundle['vehicle']['display_name']}")
    print(f"Algoritmo ganador global: {best.algorithm.upper()}")
    print(f'Costo total: {best.total_cost:,.2f} MXN')
    print(f'Valor hora: {best.hourly_value_mxn:,.2f} MXN/h')
    print(f'Seed: {best.seed}')
    print('GIF GA iterativo:', args.output_dir / 'ruta_ga_iteraciones_mapa_real.gif')
    print('Archivos generados en:', args.output_dir)


if __name__ == '__main__':
    main()
