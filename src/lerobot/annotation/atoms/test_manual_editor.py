"""Integration checks use temporary review copies; corpus and real reviews are read-only.

Run (needs the corpus and review tree on disk; set DIVERSE_DATASET_ROOT / ATOMS_REVIEW_ROOT / ATOMS_WORK_ROOT):
    uv run python -m lerobot.annotation.atoms.test_manual_editor
"""
import copy
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer

from lerobot.annotation.atoms.atoms_common import REVIEW
from lerobot.annotation.atoms.manual_editor import Store, make_handler

EPISODE = 'robochallenge__press_the_button__ep000645'


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'agent_reviews').mkdir()
        shutil.copy(REVIEW / 'agent_reviews' / f'{EPISODE}.json', self.root / 'agent_reviews')
        self.store = Store(self.root)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.store))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def test_drafts_completion_history_and_conflicts(self):
        data = self.store.load(EPISODE)
        self.assertEqual(data['errors'], [])
        review = copy.deepcopy(data['review'])
        review['parents'][0]['atoms'][0]['object'] = None
        payload = dict(review=review, revision=data['revision'], complete=True)
        result = self.store.save(EPISODE, payload)
        self.assertFalse(result['saved'])
        self.assertFalse((self.root / 'manual_overrides').exists())
        payload['complete'] = False
        draft = self.store.save(EPISODE, payload)
        self.assertTrue(draft['saved'])
        self.assertTrue(self.store.load(EPISODE)['draft'])
        with self.assertRaises(FileExistsError):
            self.store.save(EPISODE, payload)
        payload.update(review=data['review'], revision=draft['revision'], complete=True)
        result = self.store.save(EPISODE, payload)
        self.assertTrue(result['saved'])
        loaded = self.store.load(EPISODE)
        self.assertFalse(loaded['draft'])
        self.assertEqual(loaded['errors'], [])
        self.assertIn('Human', loaded['review']['reviewer'])
        payload['revision'] = result['revision']
        self.store.save(EPISODE, payload)
        self.assertEqual(len(list((self.root / 'manual_history' / EPISODE).glob('*.json'))), 1)
        self.assertEqual(loaded['sources'][0], 'manual override')

    def test_water_as_noun_and_action(self):
        from lerobot.annotation.atoms.manual_editor import validator
        for verb, obj, destination in (
            ('grasp', 'the water bottle', None),
            ('move', 'the water bottle', 'the table'),
            ('lift', 'the bottle water', None),
            ('release', 'the water bottle', 'the table'),
            ('move', 'the cup', 'the water bottle'),
            ('grasp', 'the bowl of water', None),
            ('water', 'the plant', None),
        ):
            with self.subTest(verb=verb, obj=obj, destination=destination):
                atom = dict(verb=verb, object=obj, container=destination)
                atom['subtask'] = validator.render_subtask(atom)
                self.assertEqual(validator.subtask_errors(atom, 'test'), [])
        for obj in ('the bottle and water the plant', 'the bottle to water the plant'):
            atom = dict(verb='grasp', object=obj)
            atom['subtask'] = validator.render_subtask(atom)
            errors = validator.subtask_errors(atom, 'test')
            self.assertTrue(any("second verb 'water'" in e for e in errors))

    def test_long_mistakes_allowed_within_atom(self):
        review = self.store.load(EPISODE)['review']
        parent = review['parents'][0]
        atom = parent['atoms'][0]
        atom['end_timestep_exclusive'] = parent['atoms'][-1]['end_timestep_exclusive']
        parent['atoms'] = [atom]
        atom.update(quality=1, quality_note='Sustained unsuccessful contact.')
        event = dict(start_s=6.1, end_s=15.0, kind='failed_close',
                     note='The grasp attempt remains unsuccessful throughout.')
        atom['new_mistake_events'] = [event]
        self.assertEqual(self.store.errors(EPISODE, review), [])
        event['end_s'] = 6.2
        self.assertTrue(any('shorter than 0.3 s' in e for e in self.store.errors(EPISODE, review)))
        event['end_s'] = 30.0
        self.assertTrue(any('must lie inside the atom' in e for e in self.store.errors(EPISODE, review)))

    def test_invalid_boundaries_and_duplicate_parents(self):
        original = self.store.load(EPISODE)['review']
        for mutate in (
            lambda r: r['parents'][0]['atoms'][1].update(start_timestep=341),
            lambda r: r['parents'][0]['atoms'][0].update(end_timestep_exclusive=340.5),
            lambda r: r['parents'].append(copy.deepcopy(r['parents'][0])),
            lambda r: r['parents'][0]['atoms'][0].update(verb='do everything'),
        ):
            review = copy.deepcopy(original)
            mutate(review)
            self.assertTrue(self.store.errors(EPISODE, review))

    def test_video_ranges_and_restricted_paths(self):
        url = self.url + '/video?id=' + EPISODE + '&camera=global'
        for span, length in [('bytes=0-99', 100), ('bytes=-25', 25)]:
            req = urllib.request.Request(url, headers={'Range': span})
            with urllib.request.urlopen(req) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(len(response.read()), length)
                self.assertIn('bytes', response.headers['Content-Range'])
        for bad in ['/video?id=' + EPISODE + '&camera=../../etc/passwd', '/api/episode?id=../../etc/passwd']:
            with self.assertRaises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(self.url + bad)
            self.assertEqual(exc.exception.code, 400)
        req = urllib.request.Request(url, headers={'Range': 'bytes=999999999999-'})
        with self.assertRaises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        self.assertEqual(exc.exception.code, 416)
        req = urllib.request.Request(self.url + '/api/save', data=b'{}', headers={
            'Content-Type': 'application/json', 'Origin': 'https://example.org'})
        with self.assertRaises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        self.assertEqual(exc.exception.code, 403)

    def test_browser_edit_save_reload(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=p.chromium.executable_path, headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1050})
            errors = []
            page.on('pageerror', lambda exc: errors.append(str(exc)))
            page.goto(self.url + '/#' + EPISODE)
            page.wait_for_function("document.querySelector('#task').textContent.includes('ep000645')")
            page.wait_for_function("document.querySelector('#video1').readyState >= 2 && document.querySelector('#video2').readyState >= 2")
            page.wait_for_function("Math.abs(document.querySelector('#video1').currentTime - 6) < 0.05")
            original_count = page.locator('#atoms tr').count()
            page.locator('#frame').fill('240')
            page.locator('#frame').press('Enter')
            page.locator('#frame').blur()
            page.wait_for_function("Math.abs(document.querySelector('#video1').currentTime - 8) < 0.05")
            page.locator('#split').click()
            self.assertEqual(page.locator('#atoms tr').count(), original_count + 1)
            self.assertEqual(page.locator('#start').input_value(), '240')
            page.locator('#undo').click()
            self.assertEqual(page.locator('#atoms tr').count(), original_count)
            page.locator('#atoms tr').first.click()
            page.locator('#object').fill('the blue button')
            page.locator('#end').fill('345')
            page.locator('#end').blur()
            self.assertIn('345', page.locator('#atoms tr').nth(1).inner_text())
            page.locator('#save').click()
            page.wait_for_function("document.querySelector('#status').textContent.startsWith('Draft saved.')")
            page.reload()
            page.wait_for_function("document.querySelector('#task').textContent.includes('ep000645')")
            self.assertEqual(page.locator('#end').input_value(), '345')
            page.locator('#complete').click()
            page.wait_for_function("document.querySelector('#status').textContent.startsWith('Episode reviewed.')")
            result = self.store.load(EPISODE)
            self.assertEqual(result['errors'], [])
            self.assertFalse(result['draft'])
            self.assertEqual(result['review']['parents'][0]['atoms'][1]['start_timestep'], 345)
            page.locator('#frame').fill('210')
            page.locator('#frame').blur()
            page.wait_for_function("Math.abs(document.querySelector('#video1').currentTime - 7) < 0.05")
            page.locator('#forward').click()
            page.wait_for_function("document.querySelector('#frame').value === '211'")
            page.locator('#play').click()
            page.wait_for_function("document.querySelector('#video1').currentTime > 7.2")
            page.locator('#play').click()
            sync = page.evaluate("Math.abs(document.querySelector('#video1').currentTime-document.querySelector('#video2').currentTime)")
            self.assertLess(sync, 0.2)
            page.screenshot(path='/tmp/atomic-editor-smoke.png', full_page=True)
            self.assertEqual(errors, [])
            browser.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
