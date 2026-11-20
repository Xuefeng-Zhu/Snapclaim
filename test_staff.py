"""Staff management contract. Model calls are stubbed only for these unit tests."""
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from test_search import snapclaim


class StaffTests(unittest.TestCase):
    def setUp(self):
        self.id = str(uuid4())
        self.item = {'id': self.id, 'title': 'Bottle', 'category': 'bottle', 'public_text': 'Black bottle',
                     'private_details': 'Private clue', 'proof_question': 'What is on the bottom?',
                     'found_where': 'Stage', 'finder_note': '', 'photo_key': 'private', 'status': 'open',
                     'created_at': '2026-11-20T00:00:00+00:00', 'embedding': [1, 0]}
        for ctx in (patch.object(snapclaim, 'DATABASE', False), patch.object(snapclaim, 'DESK_PIN', 'test-desk-code'),
                    patch.dict(snapclaim.MEMORY, {'items': {self.id: self.item}, 'claims': []}),
                    patch.object(snapclaim, 'embed', return_value=[1, 0])):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.client = TestClient(snapclaim.app)
        self.addCleanup(self.client.close)
        self.auth = {'X-Desk-Pin': 'test-desk-code'}

    def desk(self, **query):
        res = self.client.get('/desk', headers=self.auth, params=query)
        self.assertEqual(res.status_code, 200, res.text)
        return res.json()

    def status(self, state, revision, headers=None):
        return self.client.post(f'/desk/items/{self.id}/status', json={'status': state, 'revision': revision},
                                headers=self.auth if headers is None else headers)

    def test_ops_are_staff_only(self):
        self.assertEqual(self.status('archived', '0', {}).status_code, 401)
        res = self.client.patch(f'/desk/items/{self.id}', json={'title': 'Updated', 'public_text': 'Blue bottle',
                                'found_where': 'Desk', 'revision': '0'})
        self.assertEqual(res.status_code, 401)
        self.assertEqual(self.item['status'], 'open')
        snapclaim.embed.assert_not_called()

    def test_desk_search_filters_before_pagination_and_ignores_private_fields(self):
        self.item.update(title='Silver 100% bottle', found_where='Cedar Hall', status='archived')
        for n in range(51):
            key = str(uuid4())
            snapclaim.MEMORY['items'][key] = {**self.item, 'id': key, 'title': 'Other item',
                'found_where': 'Stage', 'status': 'open', 'created_at': f'2026-11-21T00:00:{n:02d}+00:00'}
        self.assertNotIn(self.id, [r['id'] for r in self.desk()])
        for q in (' silver ', 'CEDAR hALL', 'black bottle', '100%', self.id):
            matches = self.desk(q=q, status='archived')
            self.assertEqual([r['id'] for r in matches], [self.id])
        self.assertEqual(self.desk(q='100%', status='open'), [])
        self.assertEqual(self.desk(q='Private clue'), [])
        self.assertEqual(self.desk(q='What is on the bottom?'), [])
        self.assertEqual(self.desk(q='silver', offset=50), [])
        self.assertEqual(self.client.get('/desk', params={'q': 'silver'}).status_code, 401)
        snapclaim.embed.assert_not_called()

    def test_edit_reindexes_and_rejects_stale_edit(self):
        before = self.desk()[0]
        self.assertNotIn('private_details', before)
        data = {'title': 'Blue bottle', 'public_text': 'Steel bottle with flip lid', 'found_where': 'Cedar',
                'revision': before['revision']}
        res = self.client.patch(f'/desk/items/{self.id}', json=data, headers=self.auth)
        self.assertEqual(res.status_code, 200, res.text)
        snapclaim.embed.assert_called_once_with('Blue bottle. bottle. Steel bottle with flip lid Found at: Cedar')
        self.assertEqual(self.desk()[0]['found_where'], 'Cedar')
        self.assertEqual(self.item['private_details'], 'Private clue')
        stale = self.client.patch(f'/desk/items/{self.id}', json=data, headers=self.auth)
        self.assertEqual(stale.status_code, 409)
        snapclaim.embed.assert_called_once()

    def test_verification_is_staff_only_and_scoped_to_the_item(self):
        path = f'/desk/items/{self.id}/verification'
        self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.get(path, headers={'X-Desk-Pin': 'wrong'}).status_code, 401)
        self.assertEqual(self.client.get(f'/desk/items/{uuid4()}/verification', headers=self.auth).status_code, 404)
        initial = self.client.get(path, headers=self.auth)
        self.assertEqual(initial.status_code, 200, initial.text)
        self.assertEqual(initial.headers['cache-control'], 'no-store')
        self.assertEqual(initial.json(), {'proof_question': 'What is on the bottom?',
                                          'private_details': 'Private clue', 'attempts': [], 'has_more': False})
        with patch.object(snapclaim, 'judge_claim', return_value={'passed': False, 'reason': 'Private clue'}):
            res = self.client.post('/claim', json={'item_id': self.id, 'answer': '  Green triangle  '})
            self.assertEqual(res.status_code, 200, res.text)
            self.assertNotIn('Private clue', res.text)
        with patch.object(snapclaim, 'judge_claim', return_value={'passed': True, 'reason': 'Private clue'}):
            res = self.client.post('/claim', json={'item_id': self.id, 'answer': 'Gold circle'})
            self.assertEqual(res.status_code, 200, res.text)
            self.assertNotIn('Private clue', res.text)
        snapclaim.MEMORY['claims'].append({'item_id': str(uuid4()), 'answer': 'Unrelated answer', 'passed': True})
        self.assertEqual(self.client.get(path, headers=self.auth).json()['attempts'],
                         [{'answer': 'Gold circle', 'passed': True}, {'answer': 'Green triangle', 'passed': False}])
        self.assertTrue(all('private_details' not in r and 'attempts' not in r for r in self.desk()))
        self.item['status'] = 'open'
        public = self.client.post('/search', json={'query': 'bottle'}).json()['results'][0]
        self.assertEqual(public['proof_question'], 'What is on the bottom?')
        self.assertNotIn('private_details', public)
        self.assertNotIn('Green triangle', str(public))

    def test_verification_limits_results_and_can_page_older_attempts(self):
        snapclaim.MEMORY['claims'].extend({'item_id': self.id, 'answer': str(n), 'passed': False}
                                       for n in range(51))
        path = f'/desk/items/{self.id}/verification'
        first = self.client.get(path, headers=self.auth).json()
        second = self.client.get(path, headers=self.auth, params={'offset': 50}).json()
        self.assertEqual([r['answer'] for r in first['attempts']], [str(n) for n in range(50, 0, -1)])
        self.assertTrue(first['has_more'])
        self.assertEqual(second['attempts'], [{'answer': '0', 'passed': False}])
        self.assertFalse(second['has_more'])

    def test_archive_filter_return_reopen_and_preserve_history(self):
        snapclaim.MEMORY['claims'].append({'item_id': self.id, 'passed': False})
        before = self.desk()[0]['revision']
        self.assertEqual(self.status('archived', before).status_code, 200)
        self.assertEqual(self.status('returned', before).status_code, 409)
        self.assertEqual(self.desk(status='open'), [])
        archived = self.desk(status='archived')[0]
        self.assertEqual(archived['claims'], 1)
        self.assertEqual(self.client.post('/search', json={'query': 'bottle'}).json()['results'], [])
        self.assertEqual(self.client.post('/claim', json={'item_id': self.id, 'answer': 'a mark'}).status_code, 409)
        reopened = self.status('open', archived['revision']).json()
        self.assertEqual(reopened['status'], 'open')
        self.assertEqual(len(self.client.post('/search', json={'query': 'bottle'}).json()['results']), 1)
        self.assertEqual(self.status('returned', reopened['revision']).json()['status'], 'returned')
        self.assertEqual(self.desk(status='returned')[0]['claims'], 1)

    def test_claim_in_flight_cannot_override_staff_return(self):
        def model_response(*_):
            self.assertEqual(self.status('returned', self.desk()[0]['revision']).status_code, 200)
            return {'passed': True, 'reason': 'Specific detail provided.'}
        with patch.object(snapclaim, 'judge_claim', side_effect=model_response):
            res = self.client.post('/claim', json={'item_id': self.id, 'answer': 'a mark'})
        self.assertEqual(res.status_code, 409)
        self.assertNotIn('photo_url', res.json())
        self.assertEqual(self.item['status'], 'returned')
        self.assertEqual(snapclaim.MEMORY['claims'], [])
