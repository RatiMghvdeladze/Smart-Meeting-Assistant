import pytest
import os
import sys

# Add the project's root directory ('final') to the path
# This allows `from backend.app` to work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app import app as flask_app

@pytest.fixture
def client():
    """Create and configure a new app instance for each test."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_index_route_loads(client):
    """Test that the main page loads and contains the title."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"KIU Smart Meeting Assistant" in response.data

def test_get_meetings_endpoint(client):
    """Test that the /meetings endpoint returns a JSON list."""
    response = client.get('/meetings')
    assert response.status_code == 200
    assert response.is_json
    assert isinstance(response.get_json(), list)

def test_search_endpoint_with_empty_query(client):
    """Test that the search endpoint handles an empty query gracefully."""
    response = client.post('/search', json={'query': ''})
    assert response.status_code == 200
    assert response.is_json
    json_data = response.get_json()
    assert 'results' in json_data
    assert json_data['results'] == []