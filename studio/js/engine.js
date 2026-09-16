// Promise API over the engine worker.

let worker = null;
let seq = 0;
const pending = new Map();

function start() {
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    const entry = pending.get(data.id);
    if (!entry) return;
    pending.delete(data.id);
    if (data.ok) entry.resolve({ result: data.result, ms: data.ms });
    else {
      if (data.fatal) restart();
      entry.reject(new Error(data.fatal ? `The engine crashed: ${data.error}` : data.error));
    }
  };
  worker.onerror = (event) => {
    event.preventDefault();
    const message = event.message || "The engine failed to load (is pkg/ built? see studio/README.md)";
    for (const entry of pending.values()) entry.reject(new Error(message));
    pending.clear();
    restart();
  };
}

function restart() {
  if (worker) worker.terminate();
  worker = null;
}

export function call(op, payload) {
  if (!worker) start();
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, payload });
  });
}

export const engineVersion = () => call("version").then((r) => r.result);
export const inspectCsv = (csv) => call("inspect", csv).then((r) => r.result);
export const fitModel = (request) => call("fit", request);
