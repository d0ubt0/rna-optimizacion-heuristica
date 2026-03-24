from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .types import SolverResult


def _mexico_outline_segments() -> List[np.ndarray]:
    # Coarse outlines for mainland and Baja California; enough for route context.
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


def _route_coords(route: Sequence[int], capitals: Sequence[Dict]) -> np.ndarray:
    return np.array([[capitals[idx]['lon'], capitals[idx]['lat']] for idx in route], dtype=float)


def _build_base_axes(capitals: Sequence[Dict], figsize: Iterable[float]) -> tuple:
    fig, ax = plt.subplots(figsize=tuple(figsize))

    ax.set_facecolor('#f5fbff')
    fig.patch.set_facecolor('white')

    for segment in _mexico_outline_segments():
        ax.plot(segment[:, 0], segment[:, 1], color='#8aa4b8', linewidth=1.0, alpha=0.9)

    lon = [c['lon'] for c in capitals]
    lat = [c['lat'] for c in capitals]
    labels = [c['capital'] for c in capitals]

    ax.scatter(lon, lat, s=30, color='#12436d', alpha=0.9, zorder=3)
    for x, y, label in zip(lon, lat, labels):
        ax.text(x + 0.12, y + 0.08, label, fontsize=6.5, color='#0f2940', alpha=0.85)

    ax.set_xlim(-118.6, -86.5)
    ax.set_ylim(14.0, 33.6)
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('Recorrido del vendedor por capitales de Mexico')
    ax.grid(alpha=0.15)

    return fig, ax


def create_route_animation(
    capitals: Sequence[Dict],
    result: SolverResult,
    output_gif: Path,
    output_png: Path,
    fps: int = 6,
    figsize: Iterable[float] = (10, 8),
) -> None:
    history_routes = result.history_best_route
    history_costs = result.history_best_cost
    if not history_routes:
        history_routes = [result.route]
        history_costs = [result.total_cost]

    fig, ax = _build_base_axes(capitals, figsize)

    route_line, = ax.plot([], [], color='#d7263d', linewidth=2.0, zorder=4)
    route_points = ax.scatter([], [], color='#d7263d', s=18, zorder=5)
    info = ax.text(
        0.02,
        0.98,
        '',
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.78),
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
        f'Mejor solucion global ({result.algorithm.upper()})\n'
        f'Costo: {result.total_cost:,.2f} MXN\n'
        f'Valor hora: {result.hourly_value_mxn:,.2f} MXN/h\n'
        f'Seed: {result.seed}'
    )
    fig.savefig(output_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
