"""Local atomic-subtask editor. Run from the workspace with uv run python <this file>."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from lerobot.annotation.atoms.atoms_common import CORPUS, REVIEW, WORK, EXTERNAL_CAMERA, WRIST_CAMERA, load_corpus, read_jsonl
from lerobot.annotation.atoms.verdict import VERBS, PREPOSITIONS, check_verdict, skeleton, validator


class Store:
    def __init__(self, review_dir=REVIEW):
        self.review = Path(review_dir)
        self.episodes, self.parents = load_corpus()
        self.proposals = {}
        for p in read_jsonl(WORK / 'proposals.jsonl'):
            self.proposals.setdefault(p['episode_id'], {})[p['parent_interval_index']] = p
        self.lock = threading.RLock()

    def path(self, folder, eid):
        if eid not in self.proposals:
            raise ValueError('Unknown episode')
        return self.review / folder / f'{eid}.json'

    def revision(self, eid):
        digest = hashlib.sha256()
        for folder in ('agent_reviews', 'manual_overrides', 'manual_drafts'):
            path = self.path(folder, eid)
            digest.update(folder.encode())
            digest.update(path.read_bytes() if path.exists() else b'')
        return digest.hexdigest()

    def errors(self, eid, review):
        errors = []
        if not isinstance(review, dict) or review.get('episode_id') != eid:
            return ['Episode ID mismatch']
        parents = review.get('parents')
        if not isinstance(parents, list):
            return ['Parents must be a list']
        seen = set()
        for p in parents:
            if not isinstance(p, dict) or type(p.get('parent_interval_index')) is not int:
                return ['Parent indices must be integers']
            idx = p['parent_interval_index']
            if idx in seen:
                errors.append(f'Duplicate parent {idx}')
            seen.add(idx)
            if not isinstance(p.get('atoms'), list):
                return ['Atoms must be a list']
            for a in p['atoms']:
                if not isinstance(a, dict) or any(type(a.get(k)) is not int for k in ('start_timestep', 'end_timestep_exclusive')):
                    return ['Atom frame boundaries must be integers']
        try:
            errors += check_verdict(review, self.proposals[eid], self.episodes[eid],
                                    {p['interval_index']: p for p in self.parents[eid]})
            for p in parents:
                for a in p['atoms']:
                    errors += validator.subtask_errors(dict(a, subtask=validator.render_subtask(a)), eid)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            errors.append(f'Invalid annotation fields: {exc}')
        return errors

    def load(self, eid):
        self.path('manual_drafts', eid)  # validate ID before any filesystem access
        review = skeleton(eid)
        sources = {p['parent_interval_index']: 'proposal' for p in review['parents']}
        merged = {p['parent_interval_index']: p for p in review['parents']}
        images = []
        for folder, source in (('agent_reviews', 'existing review'), ('manual_overrides', 'manual override')):
            path = self.path(folder, eid)
            if path.exists():
                saved = json.loads(path.read_text())
                review.update({k: v for k, v in saved.items() if k != 'parents'})
                images.extend(saved.get('images', []))
                for p in saved.get('parents', []):
                    merged[p['parent_interval_index']] = p
                    sources[p['parent_interval_index']] = source
        review['parents'] = [merged[i] for i in sorted(self.proposals[eid])]
        review['images'] = list(dict.fromkeys(images))
        draft = self.path('manual_drafts', eid)
        if draft.exists():
            review = json.loads(draft.read_text())
        ep = self.episodes[eid]
        videos = CORPUS / ep['directory'] / 'videos'
        cameras = [p.stem for p in sorted(videos.glob('*.mp4'))]
        return dict(episode=ep, review=review, revision=self.revision(eid), sources=sources,
                    draft=draft.exists(), proposals=list(self.proposals[eid].values()),
                    cameras=cameras, external=EXTERNAL_CAMERA.get(ep['source'], cameras[0] if cameras else ''),
                    wrist=WRIST_CAMERA.get(ep['source'], ''), errors=self.errors(eid, review))

    def catalog(self):
        result = []
        for eid in self.proposals:
            data = self.load(eid)
            sources = list(data['sources'].values())
            status = ('draft' if data['draft'] else 'unreviewed' if 'proposal' in sources else
                      'needs fixes' if data['errors'] else 'manual' if 'manual override' in sources else 'reviewed')
            ep = data['episode']
            result.append(dict(id=eid, source=ep['source'], task=ep['task'], status=status))
        return result

    def save(self, eid, payload):
        with self.lock:
            if payload.get('revision') != self.revision(eid):
                raise FileExistsError('This episode changed on disk. Reload before saving; your browser copy is retained.')
            review = copy.deepcopy(payload['review'])
            # Drafts may fail semantic checks, but must remain loadable by the editor.
            if not isinstance(review, dict) or review.get('episode_id') != eid:
                raise ValueError('Episode ID mismatch')
            parents = review.get('parents')
            if not isinstance(parents, list) or len(parents) != len(self.proposals[eid]):
                raise ValueError('Draft must contain every parent exactly once')
            seen = set()
            for parent in parents:
                if not isinstance(parent, dict):
                    raise ValueError('Invalid parent')
                idx = parent.get('parent_interval_index')
                if type(idx) is not int or idx not in self.proposals[eid] or idx in seen:
                    raise ValueError('Invalid or duplicate parent index')
                seen.add(idx)
                atoms = parent.get('atoms')
                if not isinstance(atoms, list) or not atoms:
                    raise ValueError('Each parent needs at least one atom')
                for atom in atoms:
                    if not isinstance(atom, dict) or any(type(atom.get(k)) is not int for k in
                                                        ('start_timestep', 'end_timestep_exclusive')):
                        raise ValueError('Atom boundaries must be integer frames')
            errors = self.errors(eid, review)
            complete = payload.get('complete') is True
            if complete and errors:
                return dict(saved=False, errors=errors)
            if not isinstance(review, dict) or review.get('episode_id') != eid:
                raise ValueError('Episode ID mismatch')
            review['reviewer'] = 'Human visual review (manual editor)'
            review.setdefault('images', [])  # collector compatibility; evidence is recorded as videos below
            review['video_evidence'] = [str(p.relative_to(CORPUS)) for p in
                                       (CORPUS / self.episodes[eid]['directory'] / 'videos').glob('*.mp4')]
            review['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            dest = self.path('manual_overrides' if complete else 'manual_drafts', eid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                history = self.review / 'manual_history' / eid
                history.mkdir(parents=True, exist_ok=True)
                (history / f'{time.time_ns()}_{dest.parent.name}.json').write_bytes(dest.read_bytes())
            tmp = dest.with_suffix('.json.tmp')
            with tmp.open('w') as stream:
                json.dump(review, stream, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, dest)
            if complete:
                self.path('manual_drafts', eid).unlink(missing_ok=True)
            return dict(saved=True, errors=errors, revision=self.revision(eid), path=str(dest), complete=complete)


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def json_response(self, data, status=200):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == '/':
                    body = Path(__file__).with_suffix('.html').read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == '/api/episodes':
                    self.json_response(dict(episodes=store.catalog(), verbs=sorted(VERBS), prepositions=sorted(PREPOSITIONS)))
                elif url.path == '/api/episode':
                    self.json_response(store.load(query['id'][0]))
                elif url.path == '/video':
                    eid, camera = query['id'][0], query['camera'][0]
                    store.path('manual_drafts', eid)
                    directory = CORPUS / store.episodes[eid]['directory'] / 'videos'
                    allowed = {p.stem: p for p in directory.glob('*.mp4')}
                    if camera not in allowed:
                        raise ValueError('Unknown camera')
                    self.send_video(allowed[camera])
                else:
                    self.json_response({'error': 'Not found'}, 404)
            except (ValueError, KeyError) as exc:
                self.json_response({'error': str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_video(self, path):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            range_header = self.headers.get('Range')
            if range_header:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header)
                if not match or not any(match.groups()):
                    return self.bad_range(size)
                a, b = match.groups()
                start, end = (int(a), min(int(b), size - 1) if b else size - 1) if a else (max(0, size - int(b)), size - 1)
                if start >= size or end < start:
                    return self.bad_range(size)
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def bad_range(self, size):
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def do_POST(self):
            # Browser writes must originate in this local editor, with JSON.
            origin = self.headers.get('Origin')
            if ((origin and origin != 'http://' + self.headers.get('Host', '')) or
                    self.headers.get_content_type() != 'application/json'):
                return self.json_response({'error': 'Invalid write origin or content type'}, 403)
            try:
                if urlparse(self.path).path != '/api/save':
                    return self.json_response({'error': 'Not found'}, 404)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2_000_000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                self.json_response(store.save(payload['review']['episode_id'], payload))
            except FileExistsError as exc:
                self.json_response({'error': str(exc)}, 409)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.json_response({'error': str(exc)}, 400)

        def log_message(self, fmt, *args):
            if not self.path.startswith('/video'):
                super().log_message(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--review-dir', type=Path, default=REVIEW, help='Alternate review directory, e.g. for testing')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(Store(args.review_dir)))
    print(f'Atomic subtask editor: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
