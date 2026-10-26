// Run with: node --test lerobot/tests/scripts/test_compare_probes_model.cjs
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { runInNewContext } = require('node:vm');

const source = readFileSync(resolve(__dirname, '../../src/lerobot/scripts/_compare_probes/model.js'), 'utf8');

function viewer(hash = '', explicit = false) {
  const events = {};
  const location = { hash };
  const window = {};
  runInNewContext(source, {
    window, location, URLSearchParams, console,
    history: { replaceState: (_, __, url) => { location.hash = url; } },
    addEventListener: (event, callback) => { events[event] = callback; },
  });
  const model = window.ProbeCompare;
  model.data = {
    groups: ['Actions'],
    runs: [400, 800].map((initial_step, i) => ({
      run: { initial_step, explicit_step: explicit, label: `Run ${i}` },
      steps: [{ step: 400 }, { step: 800 }],
      probes: [
        { id: 'actions', title: 'Actions', group: 'Actions', metrics: [], panels: {
          400: [{ file: 'plot.png', kind: 'image', primary: true }],
          800: [{ file: 'plot.png', kind: 'image', primary: true }],
        } },
        { id: 'sim', title: 'Simulation', group: 'Actions', metrics: [], panels: {
          400: [{ file: 'ep0/videos/ep0_top.mp4', kind: 'video', primary: true }],
          800: [{ file: 'ep0/videos/ep0_top.mp4', kind: 'video', primary: true }],
        } },
        { id: 'old_jacobian', title: 'Jacobian', group: 'Actions', metrics: [], panels: { 400: [], 800: [] } },
      ],
    })),
  };
  model.emit();
  return { model, location, events };
}
const values = value => JSON.parse(JSON.stringify(value));

test('explicit checkpoints open aligned and whole-run inputs open the map', () => {
  const { model } = viewer('', true);
  assert.equal(model.view, 'aligned');
  assert.deepEqual(values(model.steps), [400, 800]);
  assert.equal(viewer().model.view, 'map');
  assert.equal(viewer('#view=map', true).model.view, 'map');
});

test('URL state restores the selected view, baseline, metric, and locked steps', () => {
  const { model, location } = viewer('#view=aligned&steps=800,800&baseline=1&locked=1&metric=loss&all_metrics=1&shared=1');
  assert.equal(model.view, 'aligned');
  assert.equal(model.baseline, 1);
  assert.equal(model.metric, 'loss');
  assert.equal(model.allMetrics, true);
  assert.equal(model.shared, true);
  assert.equal(model.locked, true);
  assert.deepEqual(values(model.steps), [800, 800]);
  assert.equal(new URLSearchParams(location.hash.slice(1)).get('view'), 'aligned');
});

test('same-tab links restore state and discard a lock inconsistent with their checkpoints', () => {
  const { model, location, events } = viewer();
  location.hash = '#probe=sim&view=aligned&steps=400,800&locked=1&baseline=1';
  events.hashchange();
  assert.equal(model.probeId, 'sim');
  assert.equal(model.view, 'aligned');
  assert.equal(model.baseline, 1);
  assert.equal(model.locked, false);
  assert.deepEqual(values(model.steps), [400, 800]);
});

test('common checkpoint updates validate all slots and notify observers once', () => {
  const { model } = viewer();
  const observations = [];
  model.subscribe(() => observations.push(values(model.steps)));
  assert.equal(model.setSteps([800, 800]), true);
  assert.deepEqual(observations, [[800, 800]]);
  assert.equal(model.setSteps([400, 999]), false);
  assert.equal(model.setSteps([400]), false);
  assert.deepEqual(values(model.steps), [800, 800]);
  assert.equal(observations.length, 1);
});

test('independent checkpoint updates keep the other slot unchanged', () => {
  const { model } = viewer('', true);
  model.setStep(0, 800);
  assert.deepEqual(values(model.steps), [800, 800]);
  model.setStep(1, 400);
  assert.deepEqual(values(model.steps), [800, 400]);
});

test('simulation shortcut opens playable aligned evidence and archive is opt-in', () => {
  const { model } = viewer();
  assert.equal(model.probes().some(p => p.id === 'old_jacobian'), false);
  model.setArchived(true);
  assert.equal(model.probes().some(p => p.id === 'old_jacobian'), true);
  model.jumpSim();
  assert.equal(model.view, 'aligned');
  assert.equal(model.probeId, 'sim');
  assert.equal(model.panelKey, 'ep0/videos/ep0_top.mp4');
});
