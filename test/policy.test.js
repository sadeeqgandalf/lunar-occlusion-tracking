// The browser port runs the same expert and the same networks as the Python that trained and scored them.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Episode, expertChunk, fieldWorld, fromUnit, makeExpert, makeRng, singleWorld } from '../src/policy/sim.js';
import { arLogits, bcAct, cosineSchedule, cvDecode, decode, dpEps, encode, fpVelocity, unpack } from '../src/policy/policies.js';

const dir = new URL('../policy/', import.meta.url);
const maxDiff = (a, b) => Math.max(...a.flat(Infinity).map((v, i) => Math.abs(v - b.flat(Infinity)[i])));

for (const [course, weights] of [['single rock', 'weights'], ['rock field', 'weights_field']]) {
  const manifest = JSON.parse(readFileSync(new URL(`${weights}/manifest.json`, dir)));
  const ref = JSON.parse(readFileSync(new URL(`${weights}/parity.json`, dir)));
  const W = unpack(manifest, readFileSync(new URL(`${weights}/epoch_${manifest.checkpoints.at(-1)}.bin`, dir)).buffer.slice(0));

  test(`${course}: expert and tokenizer match Python`, () => {
    ref.obs.forEach((o, i) => {
      assert.ok(maxDiff(expertChunk(o, ref.side[i]), ref.expert[i]) < 1e-4, 'expert chunk');
      assert.deepEqual(encode(ref.expert[i]), ref.tokens[i]);
      assert.ok(maxDiff(decode(ref.tokens[i]), ref.decoded[i]) < 1e-4, 'decoded chunk');
    });
  });

  test(`${course}: the five networks match PyTorch on the shipped float16 weights`, () => {
    ref.obs.forEach((o, i) => {
      const x = Float32Array.from(ref.dp_x[i].flat());
      assert.ok(maxDiff(bcAct(W.bc, o), ref.bc[i]) < 1e-4, 'BC chunk');
      assert.ok(maxDiff(arLogits(W.ar, o, ref.tokens[i].slice(0, -1)), ref.ar_logits[i]) < 1e-3, 'token logits');
      assert.ok(maxDiff(Array.from(dpEps(W.dp, x, ref.dp_k[i], o)), ref.dp_eps[i].flat()) < 1e-3, 'diffusion noise');
      assert.ok(maxDiff(Array.from(fpVelocity(W.fp, x, ref.fp_t[i], o)), ref.fp_v[i].flat()) < 1e-3, 'flow velocity');
      assert.ok(maxDiff(cvDecode(W.cv, o, ref.cv_z[i]), ref.cv_chunk[i].map(fromUnit)) < 1e-4, 'CVAE decoder');   // Python stores the unit range
    });
    assert.ok(maxDiff(cosineSchedule(ref.abar.length).abar, ref.abar) < 1e-6, 'noise schedule');
  });
}

test('the expert reaches the goal closed loop with perception noise, single rock and rock field', () => {
  const rng = makeRng(9);
  for (let i = 0; i < 300; i++) {
    const sigma = 0.05 + 0.35 * rng.uniform();
    const worlds = [
      singleWorld({ rockX: 5 + 4 * rng.uniform(), rockY: (rng.uniform() - 0.5) * 2.4, rockR: 0.8 + 0.7 * rng.uniform(), goalY: (rng.uniform() - 0.5) * 3, sigma }),
      fieldWorld(rng, sigma, i % 2 === 0),
    ];
    for (const w of worlds) {
      const ep = new Episode(w, i), expert = makeExpert(makeRng(1000 + i));
      while (!ep.done) ep.step(expert);
      assert.equal(ep.status, 'goal', `world ${i}, ${w.rocks.length} rocks`);
    }
  }
});

test('in a rock field on the lane centre, the expert chooses afresh at every rock', () => {
  let same = 0, n = 0;
  for (let i = 0; i < 300; i++) {
    const ep = new Episode(fieldWorld(makeRng(i), 0.05, true), i), expert = makeExpert(makeRng(500 + i));
    while (!ep.done) ep.step(expert);
    const s = ep.passedY.slice(0, 3).map(Math.sign);
    same += (s[0] === s[1]) + (s[1] === s[2]); n += 2;
  }
  assert.ok(same / n > 0.4 && same / n < 0.6, `same side as the rock before: ${same / n}`);
});
