import requests
import pytest

from smart_money_detection.kalshi_client import (
    ApiResponse,
    KalshiApiError,
    KalshiClient,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=''):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or ''

    def json(self):
        return self._payload


def test_request_retries_on_timeout(monkeypatch):
    client = KalshiClient(max_retries=1, backoff_factor=0.0)
    calls = []

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.Timeout("timeout")
        return FakeResponse(payload={'ok': True})

    monkeypatch.setattr(client.session, 'request', fake_request)
    monkeypatch.setattr('smart_money_detection.kalshi_client.time.sleep', lambda *_: None)

    response = client._request('GET', '/endpoint')

    assert isinstance(response, ApiResponse)
    assert response.data['ok'] is True
    assert response.metrics.retries == 1
    assert len(calls) == 2


def test_request_raises_after_retry_exhaustion(monkeypatch):
    client = KalshiClient(max_retries=1, backoff_factor=0.0)
    responses = [
        FakeResponse(status_code=500, payload={'error': 'server down'}),
        FakeResponse(status_code=502, payload={'error': 'still down'}),
    ]

    def fake_request(method, url, timeout=None, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(client.session, 'request', fake_request)
    monkeypatch.setattr('smart_money_detection.kalshi_client.time.sleep', lambda *_: None)

    with pytest.raises(KalshiApiError) as err:
        client._request('GET', '/fail')

    assert err.value.status_code == 502
    assert err.value.retries == 1
    assert err.value.response_body['error'] == 'still down'


def test_request_raises_on_client_error_without_retry(monkeypatch):
    client = KalshiClient(max_retries=3, backoff_factor=0.0)

    def fake_request(method, url, timeout=None, **kwargs):
        return FakeResponse(status_code=404, payload={'message': 'missing'})

    monkeypatch.setattr(client.session, 'request', fake_request)
    monkeypatch.setattr('smart_money_detection.kalshi_client.time.sleep', lambda *_: None)

    with pytest.raises(KalshiApiError) as err:
        client._request('GET', '/missing')

    assert err.value.status_code == 404
    assert err.value.retries == 0
    assert err.value.response_body['message'] == 'missing'


def test_demo_mode_uses_injected_mock_provider(monkeypatch):
    def mock_provider(endpoint: str):
        return {'endpoint': endpoint, 'mock': True}

    client = KalshiClient(demo_mode=True, mock_response_provider=mock_provider)

    def fail_request(*args, **kwargs):
        raise AssertionError("HTTP request should not be called in demo mode")

    monkeypatch.setattr(client.session, 'request', fail_request)

    response = client._request('GET', '/anything')

    assert response.data == {'endpoint': '/anything', 'mock': True}
    assert response.status_code == 200
    assert response.metrics.retries == 0
