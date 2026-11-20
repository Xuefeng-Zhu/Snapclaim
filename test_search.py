"""Search cutoff contract, without a database or inference network call."""
import importlib.util
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location('snapclaim', Path(__file__).with_name('app.py'))
assert spec is not None and spec.loader is not None
snapclaim = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, {
    'PYTHON_DOTENV_DISABLED': '1', 'AKAMAI_BASE_URL': 'https://example.invalid/v1',
    'AKAMAI_API_KEY': 'test-only', 'CHAT_MODEL': 'test', 'VISION_MODEL': 'test',
    'EMBEDDING_MODEL': 'test', 'PGHOST': '', 'OBJ_BUCKET': '',
}, clear=True):
    spec.loader.exec_module(snapclaim)


class SearchCutoffTest(unittest.TestCase):
    def test_memory_returns_exactly_fifty_but_not_rounded_up(self):
        items = {str(d): {'id': str(d), 'status': 'open', 'embedding': [d]} for d in (0.49, 0.5, 0.5001, 0.55)}
        with patch.object(snapclaim, 'DATABASE', False), patch.object(snapclaim, 'embed', return_value=[1]), \
             patch.object(snapclaim, 'cosine_distance', side_effect=lambda _, vector: vector[0]), \
             patch.dict(snapclaim.MEMORY, {'items': items}), TestClient(snapclaim.app) as client:
            answer = client.post('/search', json={'query': 'bottle'}).json()
        self.assertEqual([(r['id'], r['match']) for r in answer['results']], [('0.49', 51), ('0.5', 50)])
        self.assertEqual(answer['near_misses'], [])

    def test_database_does_not_expose_below_cutoff_as_suggestions(self):
        from unittest.mock import MagicMock
        db = MagicMock()
        dt = datetime(2026, 11, 20, tzinfo=timezone.utc)
        db.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            ('item', 'Bottle', 'bottle', 'A bottle', None, 'A clue?', dt, 0.5001),
            ('item2', 'Keys', 'keys', 'Some keys', None, 'A clue?', dt, 0.72),
        ]
        with patch.object(snapclaim, 'DATABASE', True), patch.object(snapclaim, 'db', db), \
             patch.object(snapclaim, 'embed', return_value=[1]), TestClient(snapclaim.app) as client:
            answer = client.post('/search', json={'query': 'watch'}).json()
        self.assertEqual(answer['results'], [])
        self.assertEqual(answer['near_misses'], [])
        self.assertIn('Nothing turned in matches', answer['message'])


if __name__ == '__main__':
    unittest.main()
