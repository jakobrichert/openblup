// Runs the OpenBLUP WebAssembly engine off the main thread.
import init, { version, inspect, fit } from "../pkg/openblup_studio.js";

const ready = init();

self.onmessage = async ({ data }) => {
  const { id, op, payload } = data;
  try {
    await ready;
    const t0 = performance.now();
    let result;
    if (op === "version") result = version();
    else if (op === "inspect") result = JSON.parse(inspect(payload));
    else if (op === "fit") result = JSON.parse(fit(JSON.stringify(payload)));
    else throw new Error(`Unknown operation '${op}'`);
    self.postMessage({ id, ok: true, result, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: String((err && err.message) || err),
      // A trap (panic) leaves the instance unusable; the client restarts us.
      fatal: err instanceof WebAssembly.RuntimeError,
    });
  }
};
