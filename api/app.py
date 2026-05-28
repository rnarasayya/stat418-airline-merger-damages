import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

# Logging -- no emoji, plain ASCII only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # required -- Streamlit and API run on different URLs in deployment

# Artifact path -- configurable via env var so same code works locally and in Docker
ARTIFACT_DIR = Path(os.environ.get('ARTIFACT_DIR', '../models/artifacts'))

try:
    x_cols          = joblib.load(ARTIFACT_DIR / 'x_cols.pkl')
    het_cols        = joblib.load(ARTIFACT_DIR / 'het_cols.pkl')
    damages_scalars = joblib.load(ARTIFACT_DIR / 'damages_scalars.pkl')
    damages_df      = pd.read_parquet(ARTIFACT_DIR / 'damages_summary.parquet')
    model_comp      = pd.read_csv(ARTIFACT_DIR / 'model_comparison.csv')
    logger.info('All artifacts loaded from %s', str(ARTIFACT_DIR))
    logger.info('Damages summary: %d routes loaded', len(damages_df))
    logger.info('Global ATE: %.4f log pts (%.2f%%)',
                damages_scalars['ate_log'],
                damages_scalars['ate_pct'] * 100)
    MODELS_LOADED = True
except Exception as e:
    logger.error('Failed to load artifacts: %s', str(e))
    MODELS_LOADED = False

# Load ML model objects separately -- not called in any endpoint, but loaded for completeness.
# These may fail due to numpy version differences between training and serving environments.
try:
    dml_model = joblib.load(ARTIFACT_DIR / 'dml_model.pkl')
    cf_model  = joblib.load(ARTIFACT_DIR / 'cf_model.pkl')
    logger.info('ML model objects loaded successfully')
except Exception as e:
    logger.warning('ML model objects could not be deserialized (numpy version mismatch): %s', str(e))
    dml_model = None
    cf_model  = None


@app.before_request
def log_request():
    logger.info('Request: %s %s', request.method, request.path)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'LinearDML',
        'artifacts_loaded': MODELS_LOADED,
        'ate_pct': round(float(damages_scalars['ate_pct']) * 100, 2),
        'routes_available': len(damages_df),
    })


@app.route('/routes', methods=['GET'])
def routes():
    route_list = (
        damages_df[['route', 'total_damages', 'avg_overcharge',
                    'total_passengers', 'avg_fare_actual', 'avg_fare_but_for']]
        .sort_values('total_damages', ascending=False)
        .head(200)
    )
    # Convert to plain Python types for JSON serialization
    records = []
    for _, row in route_list.iterrows():
        records.append({
            'route':             str(row['route']),
            'total_damages':     round(float(row['total_damages']), 2),
            'avg_overcharge':    round(float(row['avg_overcharge']), 2),
            'total_passengers':  round(float(row['total_passengers']), 0),
            'avg_fare_actual':   round(float(row['avg_fare_actual']), 2),
            'avg_fare_but_for':  round(float(row['avg_fare_but_for']), 2),
        })
    return jsonify({
        'routes':        [r['route'] for r in records],
        'count':         len(records),
        'route_details': records,
    })


@app.route('/damages', methods=['GET'])
def damages():
    top_n = request.args.get('top_n', 20, type=int)
    top_n = min(max(top_n, 1), 100)  # clamp between 1 and 100

    top_routes = []
    for _, row in damages_df.head(top_n).iterrows():
        top_routes.append({
            'route':             str(row['route']),
            'total_damages':     round(float(row['total_damages']), 2),
            'avg_overcharge':    round(float(row['avg_overcharge']), 2),
            'total_passengers':  round(float(row['total_passengers']), 0),
            'avg_fare_actual':   round(float(row['avg_fare_actual']), 2),
            'avg_fare_but_for':  round(float(row['avg_fare_but_for']), 2),
        })

    return jsonify({
        'ate_pct':                   round(float(damages_scalars['ate_pct']) * 100, 4),
        'ate_log':                   round(float(damages_scalars['ate_log']), 6),
        'total_damages_bn':          round(float(damages_scalars['total_damages_bn']), 3),
        'total_passengers_mn':       round(float(damages_scalars['total_passengers_mn']), 1),
        'avg_overcharge_per_ticket': round(float(damages_scalars['avg_overcharge_per_ticket']), 2),
        'top_routes':                top_routes,
        'model_comparison':          model_comp.to_dict(orient='records'),
    })


@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True, silent=True)
    if not body or 'route' not in body:
        return jsonify({'error': 'Missing required field: route'}), 400

    route = str(body['route']).strip().upper()

    match = damages_df[damages_df['route'] == route]

    if match.empty:
        # Route not in treated set -- return global ATE with explanation
        return jsonify({
            'route':            route,
            'found_in_damages': False,
            'note':             'Route not in treated set (AA and US did not both operate this route pre-merger). Returning global Double ML ATE.',
            'overcharge_pct':   round(float(damages_scalars['ate_pct']) * 100, 4),
            'ate_pct':          round(float(damages_scalars['ate_pct']) * 100, 4),
            'ate_log':          round(float(damages_scalars['ate_log']), 6),
            'avg_fare_actual':  None,
            'avg_fare_but_for': None,
            'avg_overcharge':   None,
            'total_damages':    None,
            'total_passengers': None,
        })

    row = match.iloc[0]
    return jsonify({
        'route':            route,
        'found_in_damages': True,
        'avg_fare_actual':  round(float(row['avg_fare_actual']), 2),
        'avg_fare_but_for': round(float(row['avg_fare_but_for']), 2),
        'avg_overcharge':   round(float(row['avg_overcharge']), 2),
        'overcharge_pct':   round(float(damages_scalars['ate_pct']) * 100, 4),
        'total_damages':    round(float(row['total_damages']), 2),
        'total_passengers': round(float(row['total_passengers']), 0),
        'ate_pct':          round(float(damages_scalars['ate_pct']) * 100, 4),
        'ate_log':          round(float(damages_scalars['ate_log']), 6),
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'status': 404}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'status': 500}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
