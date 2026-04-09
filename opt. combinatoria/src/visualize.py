import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .types import SolverResult


def _mexico_outline_segments() -> List[np.ndarray]:
    # Fallback silhouette if GeoJSON is unavailable.
    mainland = np.array(
        [
            [-117.1, 32.5],
            [-114.8, 31.3],
            [-112.2, 31.3],
            [-109.1, 31.3],
            [-107.0, 30.0],
            [-105.0, 28.5],
            [-103.0, 27.5],
            [-101.0, 27.7],
            [-99.2, 26.6],
            [-97.5, 25.9],
            [-96.1, 22.5],
            [-95.3, 20.4],
            [-94.8, 18.7],
            [-93.8, 17.7],
            [-92.5, 16.7],
            [-91.5, 17.1],
            [-90.4, 18.4],
            [-88.3, 21.3],
            [-87.5, 21.6],
            [-88.3, 20.3],
            [-89.5, 19.8],
            [-91.3, 18.7],
            [-93.0, 17.5],
            [-94.7, 16.5],
            [-96.8, 15.8],
            [-98.5, 16.0],
            [-100.3, 17.2],
            [-101.7, 18.3],
            [-103.3, 18.8],
            [-105.3, 20.0],
            [-106.8, 21.8],
            [-108.6, 23.9],
            [-110.2, 25.1],
            [-112.2, 26.8],
            [-114.1, 28.4],
            [-115.8, 30.5],
            [-117.1, 32.5],
        ]
    )
    baja = np.array(
        [
            [-117.1, 32.5],
            [-116.3, 31.2],
            [-115.6, 30.0],
            [-114.8, 28.7],
            [-114.3, 27.4],
            [-113.8, 26.3],
            [-113.2, 25.1],
            [-112.6, 24.0],
            [-112.0, 23.0],
            [-111.5, 22.1],
            [-110.9, 23.2],
            [-111.2, 24.8],
            [-111.5, 26.0],
            [-112.1, 27.2],
            [-112.9, 28.5],
            [-113.8, 29.8],
            [-114.7, 31.0],
            [-116.0, 32.0],
            [-117.1, 32.5],
        ]
    )
    return [mainland, baja]


def _extract_country_rings(geojson_path: Path) -> List[np.ndarray]:
    payload = json.loads(geojson_path.read_text(encoding='utf-8'))
    features = []
    if payload.get('type') == 'FeatureCollection':
        features = payload.get('features', [])
    elif payload.get('type') == 'Feature':
        features = [payload]

    rings: List[np.ndarray] = []
    for feature in features:
        geometry = feature.get('geometry', {})
        gtype = geometry.get('type')
        coords = geometry.get('coordinates', [])

        if gtype == 'Polygon':
            if coords:
                rings.append(np.array(coords[0], dtype=float))
        elif gtype == 'MultiPolygon':
            for polygon in coords:
                if polygon:
                    rings.append(np.array(polygon[0], dtype=float))

    if not rings:
        raise ValueError(f'No polygon geometry found in {geojson_path}.')
    return rings


def _route_coords(route: Sequence[int], capitals: Sequence[Dict]) -> np.ndarray:
    return np.array([[capitals[idx]['lon'], capitals[idx]['lat']] for idx in route], dtype=float)


def _build_base_axes(
    capitals: Sequence[Dict],
    figsize: Iterable[float],
    mexico_geojson_path: Path,
) -> tuple:
    fig, ax = plt.subplots(figsize=tuple(figsize))

    ax.set_facecolor('#eef6fb')
    fig.patch.set_facecolor('white')

    try:
        segments = _extract_country_rings(mexico_geojson_path)
        for segment in segments:
            ax.fill(segment[:, 0], segment[:, 1], facecolor='#e4edf4', edgecolor='#7f95aa', linewidth=0.85, zorder=1)
    except Exception:
        segments = _mexico_outline_segments()
        for segment in segments:
            ax.plot(segment[:, 0], segment[:, 1], color='#7f95aa', linewidth=1.0, alpha=0.95)

    lon = [c['lon'] for c in capitals]
    lat = [c['lat'] for c in capitals]
    labels = [c['capital'] for c in capitals]

    ax.scatter(lon, lat, s=28, color='#0a4f7b', alpha=0.9, zorder=3)
    for x, y, label in zip(lon, lat, labels):
        ax.text(x + 0.09, y + 0.07, label, fontsize=6.2, color='#0f2940', alpha=0.87)

    all_x = np.concatenate([seg[:, 0] for seg in segments])
    all_y = np.concatenate([seg[:, 1] for seg in segments])
    x_pad = max((all_x.max() - all_x.min()) * 0.05, 0.8)
    y_pad = max((all_y.max() - all_y.min()) * 0.05, 0.6)

    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('Recorrido del vendedor por capitales de Mexico (mapa real)')
    ax.grid(alpha=0.2, linewidth=0.6)

    return fig, ax


def create_route_animation(
    capitals: Sequence[Dict],
    result: SolverResult,
    output_gif: Path,
    output_png: Path,
    mexico_geojson_path: Path,
    fps: int = 6,
    figsize: Iterable[float] = (10, 8),
) -> None:
    history_routes = result.history_best_route
    history_costs = result.history_best_cost
    if not history_routes:
        history_routes = [result.route]
        history_costs = [result.total_cost]

    fig, ax = _build_base_axes(capitals, figsize, mexico_geojson_path)

    route_line, = ax.plot([], [], color='#d7263d', linewidth=2.0, zorder=4)
    route_points = ax.scatter([], [], color='#d7263d', s=18, zorder=5)
    info = ax.text(
        0.02,
        0.98,
        '',
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.82),
    )

    def init():
        route_line.set_data([], [])
        route_points.set_offsets(np.empty((0, 2)))
        info.set_text('')
        return route_line, route_points, info

    def update(frame: int):
        route = history_routes[frame]
        coords = _route_coords(route, capitals)
        route_line.set_data(coords[:, 0], coords[:, 1])
        route_points.set_offsets(coords)

        cost = history_costs[frame]
        info.set_text(
            f'Algoritmo: {result.algorithm.upper()}\n'
            f'Iteracion: {frame + 1}/{len(history_routes)}\n'
            f'Mejor costo acumulado: {cost:,.2f} MXN\n'
            f'Valor hora: {result.hourly_value_mxn:,.2f} MXN/h\n'
            f'Seed: {result.seed}'
        )
        return route_line, route_points, info

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(history_routes),
        interval=int(1000 / max(fps, 1)),
        blit=False,
    )

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_gif, writer='pillow', fps=fps)

    final_coords = _route_coords(result.route, capitals)
    route_line.set_data(final_coords[:, 0], final_coords[:, 1])
    route_points.set_offsets(final_coords)
    info.set_text(
        f'Mejor solucion final ({result.algorithm.upper()})\n'
        f'Costo: {result.total_cost:,.2f} MXN\n'
        f'Valor hora: {result.hourly_value_mxn:,.2f} MXN/h\n'
        f'Seed: {result.seed}'
    )
    fig.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
