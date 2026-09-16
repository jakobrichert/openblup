// Smoke test for OpenBLUP Studio: the WebAssembly engine and the requests the
// UI builds agree for every built-in example. Run after `studio/build.sh`:
//
//   node studio/tests/smoke.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, version, inspect, fit } from "../pkg/openblup_studio.js";
import { EXAMPLES, fitRequest, cliCommand } from "../js/examples.js";

const read = (path) => readFileSync(new URL(path, import.meta.url));
initSync({ module: read("../pkg/openblup_studio_bg.wasm") });
console.log(`engine ${version()}`);

const checks = {
  spatial: (r) => {
    assert.equal(r.engine, "general");
    assert.equal(r.grid.row_levels.length, 12);
    assert.equal(r.grid.col_levels.length, 20);
    assert.equal(r.grid.cell_index.length, r.model.n_obs);
  },
  met: (r) => {
    assert.equal(r.covariances.length, 1);
    assert.equal(r.covariances[0].loadings.length, 4);
  },
  animal: (r) => {
    assert.equal(r.pedigree.n_animals, 63);
    assert.equal(r.fit.random_effects[0].effects.length, 63);
    assert.equal(r.cv.folds.length, 5);
  },
  rcbd: (r) => {
    assert.equal(r.model.n_obs, 17);
    assert.ok(r.diagnostics.leverage.every((h) => h >= 0 && h <= 1));
  },
};

for (const ex of EXAMPLES) {
  const data = read(`../../examples/${ex.data}`).toString();
  const pedigree = ex.pedigree ? read(`../../examples/${ex.pedigree}`).toString() : null;
  const info = JSON.parse(inspect(data));
  assert.ok(info.columns.some((c) => c.name === ex.model.response), `${ex.id}: response column`);

  const options = { algorithm: "ai", ddf: "containment", cv: 0, seed: 1, maxIter: 50, ...ex.options };
  const request = fitRequest({ data, pedigree, model: ex.model, options });
  const t0 = performance.now();
  const result = JSON.parse(fit(JSON.stringify(request)));
  const ms = performance.now() - t0;
  assert.equal(result.fit.converged, true, `${ex.id}: converged`);
  assert.equal(result.observations.y.length, result.model.n_obs);
  assert.ok(Number.isFinite(result.fit.log_likelihood));
  assert.equal(result.ddf_method, options.ddf, `${ex.id}: ddf method`);
  checks[ex.id]?.(result);
  assert.match(cliCommand({ dataName: ex.data, pedigreeName: ex.pedigree, model: ex.model, options }), /^openblup fit/);
  console.log(`ok ${ex.id.padEnd(8)} logL ${result.fit.log_likelihood.toFixed(3).padStart(10)}  ${result.fit.n_iterations} iterations  ${ms.toFixed(0)} ms`);
}

// Engine errors surface as exceptions with a readable message.
const data = read("../../examples/field_trial.csv").toString();
assert.throws(
  () => fit(JSON.stringify({ data, spec: { response: "yield", random: ["nope"] } })),
  /'nope'/,
);
assert.throws(() => inspect("a,b\n1\n"), /./);
console.log("ok errors");
