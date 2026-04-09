import argparse
import csv
import gzip
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json_response(raw: bytes) -> Dict[str, Any]:
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return json.loads(raw.decode('utf-8'))


class InegiRuteoClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_s: float,
        retries: int,
        retry_backoff_s: float,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout_s = timeout_s
        self.retries = max(1, retries)
        self.retry_backoff_s = max(0.0, retry_backoff_s)
        self.call_counts: Dict[str, int] = {'buscalinea': 0, 'optima': 0, 'combustible': 0}
        self.meta_source: Dict[str, str] = {}

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(payload)
        body['type'] = 'json'
        body['key'] = self.token

        data = urllib.parse.urlencode(body).encode('utf-8')
        request = urllib.request.Request(
            f'{self.base_url}/{endpoint.lstrip('/')}',
            data=data,
            method='POST',
            headers={
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'Accept': 'application/json',
            },
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload_json = _read_json_response(response.read())
                break
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise RuntimeError(f'INEGI request failed at endpoint {endpoint}: {exc}') from exc
                time.sleep(self.retry_backoff_s * attempt)
        else:
            raise RuntimeError(f'INEGI request failed at endpoint {endpoint}: {last_error}')

        status = payload_json.get('response', {})
        if not status.get('success', False):
            message = status.get('message', 'Unknown error')
            raise RuntimeError(f'INEGI endpoint {endpoint} responded with error: {message}')

        meta = payload_json.get('meta', {})
        source = meta.get('fuente')
        if source:
            self.meta_source[endpoint] = str(source)

        if endpoint in self.call_counts:
            self.call_counts[endpoint] += 1
        return payload_json

    def snap_to_network(self, lon: float, lat: float, projection: str, scales: List[int]) -> Dict[str, str]:
        last_error: Optional[Exception] = None
        for scale in scales:
            try:
                payload = self._post(
                    'buscalinea',
                    {
                        'proj': projection,
                        'escala': str(scale),
                        'x': f'{lon:.8f}',
                        'y': f'{lat:.8f}',
                    },
                )
                data = payload.get('data', {})
                if isinstance(data, list):
                    if not data:
                        raise RuntimeError('INEGI buscalinea returned an empty list.')
                    data = data[0]

                required = ('id_routing_net', 'source', 'target')
                missing = [name for name in required if name not in data]
                if missing:
                    raise RuntimeError(f'INEGI buscalinea missing keys: {missing}')

                return {
                    'id_routing_net': str(data['id_routing_net']),
                    'source': str(data['source']),
                    'target': str(data['target']),
                    'scale_used': str(scale),
                }
            except RuntimeError as exc:
                last_error = exc
                message = str(exc).casefold()
                no_line = ('no se encontro linea de red' in message) or ('no se encontró línea de red' in message)
                if no_line:
                    continue
                raise

        raise RuntimeError(f'Unable to snap point ({lon}, {lat}) with scales {scales}: {last_error}')

    def route_cost(self, origin: Dict[str, str], dest: Dict[str, str], vehicle_code: str, e_code: str) -> Tuple[float, float, float]:
        payload = self._post(
            'optima',
            {
                'proj': 'MERC',
                'id_i': origin['id_routing_net'],
                'source_i': origin['source'],
                'target_i': origin['target'],
                'id_f': dest['id_routing_net'],
                'source_f': dest['source'],
                'target_f': dest['target'],
                'v': vehicle_code,
                'e': e_code,
            },
        )
        data = payload.get('data', {})
        required = ('long_km', 'tiempo_min', 'costo_caseta')
        missing = [name for name in required if name not in data]
        if missing:
            raise RuntimeError(f'INEGI optima missing keys: {missing}')

        distance_km = float(data['long_km'])
        time_h = float(data['tiempo_min']) / 60.0
        tolls_mxn = float(data['costo_caseta'])
        return distance_km, time_h, tolls_mxn

    def fuel_prices(self) -> List[Dict[str, Any]]:
        payload = self._post('combustible', {})
        data = payload.get('data', [])
        if not isinstance(data, list):
            raise RuntimeError('INEGI combustible response is not a list.')
        prices: List[Dict[str, Any]] = []
        for item in data:
            prices.append(
                {
                    'tipo': str(item.get('tipo', '')).strip(),
                    'costo': float(item.get('costo', 0.0)),
                    'tipo_costo': str(item.get('tipo_costo', '')).strip(),
                }
            )
        return prices


def _load_capitals(path: Path) -> List[Dict[str, Any]]:
    capitals: List[Dict[str, Any]] = []
    with path.open(newline='', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        for idx, row in enumerate(reader):
            capitals.append(
                {
                    'idx': idx,
                    'state': row['state'],
                    'capital': row['capital'],
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                }
            )
    if not capitals:
        raise ValueError('No capitals loaded from CSV.')
    return capitals


def _sorted_rows(matrix_rows: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted(matrix_rows.keys(), key=lambda item: (item[0], item[1]))
    return [matrix_rows[key] for key in keys]


def _write_matrix_csv(path: Path, matrix_rows: Dict[Tuple[int, int], Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'origin_idx',
        'dest_idx',
        'origin_capital',
        'dest_capital',
        'distance_km',
        'time_h',
        'tolls_mxn',
    ]
    with path.open('w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_sorted_rows(matrix_rows))


def _write_provenance(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _select_fuel_price(prices: List[Dict[str, Any]], fuel_type: str) -> Optional[Dict[str, Any]]:
    normalized = fuel_type.strip().casefold()
    for item in prices:
        if item['tipo'].casefold() == normalized:
            return item
    return None


def _load_existing_matrix(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if not path.exists():
        return {}

    existing: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with path.open(newline='', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            i = int(row['origin_idx'])
            j = int(row['dest_idx'])
            parsed = {
                'origin_idx': i,
                'dest_idx': j,
                'origin_capital': row['origin_capital'],
                'dest_capital': row['dest_capital'],
                'distance_km': float(row['distance_km']),
                'time_h': float(row['time_h']),
                'tolls_mxn': float(row['tolls_mxn']),
            }
            if i == j:
                existing[(i, j)] = parsed
                continue

            if parsed['distance_km'] > 0.0 and parsed['time_h'] > 0.0:
                existing[(i, j)] = parsed
    return existing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build snapshot matrix from INEGI Ruteo API (SAKBE v3.1).')
    parser.add_argument(
        '--capitals-csv',
        type=Path,
        default=Path('opt. combinatoria/data/capitales_32.csv'),
        help='Path to capitals CSV.',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path('opt. combinatoria/data/costos_matriz.csv'),
        help='Path for generated cost matrix CSV.',
    )
    parser.add_argument(
        '--provenance-json',
        type=Path,
        default=Path('opt. combinatoria/data/costos_fuentes.json'),
        help='Path for ETL provenance metadata.',
    )
    parser.add_argument(
        '--token-env',
        default='INEGI_RUTEO_TOKEN',
        help='Environment variable containing INEGI token.',
    )
    parser.add_argument(
        '--base-url',
        default='https://gaia.inegi.org.mx/sakbe_v3.1',
        help='INEGI API base URL.',
    )
    parser.add_argument('--projection', default='GRS80', help='Projection used in buscalinea endpoint.')
    parser.add_argument('--scale', type=int, default=20000, help='Primary scale for buscalinea endpoint.')
    parser.add_argument(
        '--fallback-scales',
        default='30000,50000,100000,200000',
        help='Comma-separated fallback scales when buscalinea does not find a line.',
    )
    parser.add_argument('--vehicle-code', default='0', help='INEGI optima parameter v (vehicle type).')
    parser.add_argument('--e-code', default='0', help='INEGI optima parameter e.')
    parser.add_argument('--fuel-type', default='Regular', help='Fuel type to highlight in provenance metadata.')
    parser.add_argument('--timeout-s', type=float, default=60.0, help='Per-request timeout in seconds.')
    parser.add_argument('--retries', type=int, default=3, help='Retries per request.')
    parser.add_argument('--retry-backoff-s', type=float, default=0.6, help='Linear retry backoff step in seconds.')
    parser.add_argument('--delay-s', type=float, default=0.0, help='Delay between API calls in seconds.')
    parser.add_argument('--checkpoint-every', type=int, default=20, help='Write checkpoint every N resolved routes.')
    parser.add_argument('--ignore-existing', action='store_true', help='Ignore existing output CSV and rebuild from scratch.')
    parser.add_argument(
        '--max-routes',
        type=int,
        default=0,
        help='If >0, resolve at most this many missing non-diagonal routes.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(
            f'Environment variable {args.token_env} is required. '
            'Set it with your INEGI token before running this ETL.'
        )

    capitals = _load_capitals(args.capitals_csv)
    client = InegiRuteoClient(
        base_url=args.base_url,
        token=token,
        timeout_s=args.timeout_s,
        retries=args.retries,
        retry_backoff_s=args.retry_backoff_s,
    )

    fallback_scales: List[int] = []
    for value in str(args.fallback_scales).split(','):
        value = value.strip()
        if value:
            fallback_scales.append(int(value))
    scale_candidates = [int(args.scale)] + [s for s in fallback_scales if s != int(args.scale)]

    total_capitals = len(capitals)
    total_possible_routes = total_capitals * (total_capitals - 1)

    matrix_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for origin in capitals:
        for dest in capitals:
            i = int(origin['idx'])
            j = int(dest['idx'])
            matrix_rows[(i, j)] = {
                'origin_idx': i,
                'dest_idx': j,
                'origin_capital': origin['capital'],
                'dest_capital': dest['capital'],
                'distance_km': 0.0,
                'time_h': 0.0,
                'tolls_mxn': 0.0,
            }

    existing_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not args.ignore_existing:
        existing_rows = _load_existing_matrix(args.output_csv)
        if existing_rows:
            matrix_rows.update(existing_rows)
            print(f'Loaded {len(existing_rows)} existing rows from {args.output_csv} (resume mode).')

    snapped_nodes: Dict[int, Dict[str, str]] = {}
    for capital in capitals:
        idx = int(capital['idx'])
        snapped_nodes[idx] = client.snap_to_network(
            lon=float(capital['lon']),
            lat=float(capital['lat']),
            projection=args.projection,
            scales=scale_candidates,
        )
        scale_used = snapped_nodes[idx]['scale_used']
        print(f"[snap {idx + 1}/{total_capitals}] {capital['capital']} linked to RNC (escala {scale_used})")
        if args.delay_s > 0:
            time.sleep(args.delay_s)

    missing_pairs: List[Tuple[int, int]] = []
    for origin in capitals:
        for dest in capitals:
            i = int(origin['idx'])
            j = int(dest['idx'])
            if i == j:
                continue
            row = matrix_rows[(i, j)]
            resolved = float(row['distance_km']) > 0.0 and float(row['time_h']) > 0.0
            if not resolved:
                missing_pairs.append((i, j))

    if not missing_pairs:
        print('All non-diagonal routes are already resolved in the existing snapshot.')

    route_limit = len(missing_pairs)
    if args.max_routes > 0:
        route_limit = min(route_limit, args.max_routes)

    processed_this_run = 0
    failed_pairs: List[Tuple[int, int]] = []
    for i, j in missing_pairs[:route_limit]:
        origin = capitals[i]
        dest = capitals[j]
        try:
            distance_km, time_h, tolls_mxn = client.route_cost(
                origin=snapped_nodes[i],
                dest=snapped_nodes[j],
                vehicle_code=args.vehicle_code,
                e_code=args.e_code,
            )
        except Exception as exc:
            failed_pairs.append((i, j))
            print(f"WARN: route {origin['capital']} -> {dest['capital']} failed after retries: {exc}")
            continue

        matrix_rows[(i, j)] = {
            'origin_idx': i,
            'dest_idx': j,
            'origin_capital': origin['capital'],
            'dest_capital': dest['capital'],
            'distance_km': round(distance_km, 3),
            'time_h': round(time_h, 3),
            'tolls_mxn': round(tolls_mxn, 2),
        }

        processed_this_run += 1
        total_completed = sum(
            1
            for key, row in matrix_rows.items()
            if key[0] != key[1] and float(row['distance_km']) > 0.0 and float(row['time_h']) > 0.0
        )
        print(
            f"[route {processed_this_run}/{route_limit}] "
            f"{origin['capital']} -> {dest['capital']} "
            f"({matrix_rows[(i, j)]['distance_km']} km, {matrix_rows[(i, j)]['time_h']} h, "
            f"peaje {matrix_rows[(i, j)]['tolls_mxn']} MXN) | "
            f"completadas: {total_completed}/{total_possible_routes}"
        )
        if args.checkpoint_every > 0 and processed_this_run % args.checkpoint_every == 0:
            _write_matrix_csv(args.output_csv, matrix_rows)
            print(f'Checkpoint saved at {args.output_csv}')

        if args.delay_s > 0:
            time.sleep(args.delay_s)

    if failed_pairs:
        print(f'WARN: {len(failed_pairs)} routes failed in this run. Re-run to continue filling missing pairs.')

    _write_matrix_csv(args.output_csv, matrix_rows)

    fuel_prices = client.fuel_prices()
    selected_fuel = _select_fuel_price(fuel_prices, args.fuel_type)

    complete_expected_rows = total_capitals * total_capitals
    non_diagonal_rows = sum(1 for key in matrix_rows.keys() if key[0] != key[1])
    non_zero_distance_rows = sum(
        1
        for key, row in matrix_rows.items()
        if key[0] != key[1] and float(row['distance_km']) > 0.0 and float(row['time_h']) > 0.0
    )

    provenance = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {
            'service': 'INEGI SAKBE Sistema de Ruteo de Mexico v3.1',
            'base_url': args.base_url,
            'token_env_var': args.token_env,
            'endpoint_meta': client.meta_source,
            'endpoints_used': ['buscalinea', 'optima', 'combustible'],
        },
        'parameters': {
            'projection': args.projection,
            'scale_candidates': scale_candidates,
            'vehicle_code': args.vehicle_code,
            'e_code': args.e_code,
            'fuel_type_selected': args.fuel_type,
            'timeout_s': args.timeout_s,
            'retries': args.retries,
            'retry_backoff_s': args.retry_backoff_s,
            'delay_s': args.delay_s,
            'checkpoint_every': args.checkpoint_every,
            'ignore_existing': bool(args.ignore_existing),
            'max_routes': args.max_routes,
        },
        'matrix_summary': {
            'capitals': total_capitals,
            'rows_written': len(matrix_rows),
            'expected_rows_full': complete_expected_rows,
            'non_diagonal_rows': non_diagonal_rows,
            'expected_non_diagonal_rows_full': total_possible_routes,
            'non_zero_distance_rows': non_zero_distance_rows,
            'existing_rows_loaded': len(existing_rows),
            'routes_computed_this_run': processed_this_run,
            'failed_routes_this_run': len(failed_pairs),
            'is_complete_snapshot': non_zero_distance_rows == total_possible_routes,
        },
        'fuel_reference': {
            'selected': selected_fuel,
            'available_prices': fuel_prices,
        },
        'api_call_counts': client.call_counts,
    }
    _write_provenance(args.provenance_json, provenance)

    print('ETL finished successfully.')
    print(f'CSV: {args.output_csv}')
    print(f'Provenance: {args.provenance_json}')


if __name__ == '__main__':
    main()











