// The browser port runs the same expert and the same networks as the Python that trained and scored them.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Episode, expertChunk, makeExpert, makeRng, GOAL_X } from '../src/policy/sim.js';
import { arLogits, bcAct, cosineSchedule, decode, dpEps, encode, unpack } from '../src/policy/policies.js';

const dir = new URL('../policy/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('weights/manifest.json', dir)));
const ref = JSON.parse(readFileSync(new URL('weights/parity.json', dir)));
const last = manifest.checkpoints.at(-1);
const W = unpack(manifest, readFileSync(new URL(`weights/epoch_${last}.bin`, dir)).buffer.slice(0));

const maxDiff = (a, b) => Math.max(...a.flat(Infinity).map((v, i) => Math.abs(v - b.flat(Infinity)[i])));

test('expert and tokenizer match Python', () => {
  ref.obs.forEach((o, i) => {
    assert.ok(maxDiff(expertChunk(o, ref.side[i]), ref.expert[i]) < 1e-4, 'expert chunk');
    assert.deepEqual(encode(ref.expert[i]), ref.tokens[i]);
    assert.ok(maxDiff(decode(ref.tokens[i]), ref.decoded[i]) < 1e-4, 'decoded chunk');
  });
});

test('the three networks match PyTorch on the shipped float16 weights', () => {
  ref.obs.forEach((o, i) => {
    assert.ok(maxDiff(bcAct(W.bc, o), ref.bc[i]) < 1e-4, 'BC chunk');
    assert.ok(maxDiff(arLogits(W.ar, o, ref.tokens[i].slice(0, -1)), ref.ar_logits[i]) < 1e-3, 'token logits');
    assert.ok(maxDiff(Array.from(dpEps(W.dp, Float32Array.from(ref.dp_x[i].flat()), ref.dp_k[i], o)), ref.dp_eps[i].flat()) < 1e-3, 'diffusion noise');
  });
  assert.ok(maxDiff(cosineSchedule(ref.abar.length).abar, ref.abar) < 1e-6, 'noise schedule');
});

test('the expert reaches the goal in every world, closed loop with perception noise', () => {
  const rng = makeRng(9);
  for (let i = 0; i < 300; i++) {
    const w = { rockX: 5 + 4 * rng.uniform(), rockY: (rng.uniform() - 0.5) * 2.4, rockR: 0.8 + 0.7 * rng.uniform(),
      goalY: (rng.uniform() - 0.5) * 3, sigma: 0.05 + 0.35 * rng.uniform() };
    const ep = new Episode(w, i), expert = makeExpert(makeRng(1000 + i));
    while (!ep.done) ep.step(expert);
    assert.equal(ep.status, 'goal', `world ${i}`);
    assert.ok(ep.pose.x > GOAL_X - 2);
  }
});
