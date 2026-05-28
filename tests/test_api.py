import pytest
import json
import os
import sys

# Add api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

# Set artifact path before importing app
os.environ['ARTIFACT_DIR'] = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'artifacts'
)

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get('/health')
        assert r.status_code == 200

    def test_health_fields(self, client):
        data = json.loads(client.get('/health').data)
        assert data['status'] == 'ok'
        assert data['artifacts_loaded'] is True
        assert 'ate_pct' in data
        assert data['ate_pct'] > 0

class TestRoutes:
    def test_routes_returns_200(self, client):
        r = client.get('/routes')
        assert r.status_code == 200

    def test_routes_has_data(self, client):
        data = json.loads(client.get('/routes').data)
        assert 'routes' in data
        assert len(data['routes']) > 0

    def test_lax_jfk_in_routes(self, client):
        data = json.loads(client.get('/routes').data)
        assert 'LAX-JFK' in data['routes']

class TestDamages:
    def test_damages_returns_200(self, client):
        r = client.get('/damages')
        assert r.status_code == 200

    def test_damages_fields(self, client):
        data = json.loads(client.get('/damages').data)
        assert 'total_damages_bn' in data
        assert 'avg_overcharge_per_ticket' in data
        assert 'top_routes' in data
        assert 'model_comparison' in data
        assert data['total_damages_bn'] > 0

    def test_damages_top_n(self, client):
        data = json.loads(client.get('/damages?top_n=5').data)
        assert len(data['top_routes']) == 5

class TestPredict:
    def test_predict_known_route(self, client):
        r = client.post('/predict',
                        data=json.dumps({'route': 'LAX-JFK'}),
                        content_type='application/json')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data['found_in_damages'] is True
        assert data['avg_fare_actual'] > 0
        assert data['avg_overcharge'] > 0
        assert data['total_damages'] > 0

    def test_predict_case_insensitive(self, client):
        r = client.post('/predict',
                        data=json.dumps({'route': 'lax-jfk'}),
                        content_type='application/json')
        data = json.loads(r.data)
        assert data['route'] == 'LAX-JFK'
        assert data['found_in_damages'] is True

    def test_predict_unknown_route(self, client):
        r = client.post('/predict',
                        data=json.dumps({'route': 'ZZZ-ZZZ'}),
                        content_type='application/json')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data['found_in_damages'] is False
        assert data['ate_pct'] > 0
        assert 'note' in data

    def test_predict_missing_route_field(self, client):
        r = client.post('/predict',
                        data=json.dumps({}),
                        content_type='application/json')
        assert r.status_code == 400

    def test_predict_empty_body(self, client):
        r = client.post('/predict',
                        data='',
                        content_type='application/json')
        assert r.status_code == 400

class TestErrors:
    def test_404_endpoint(self, client):
        r = client.get('/nonexistent')
        assert r.status_code == 404
        data = json.loads(r.data)
        assert 'error' in data
