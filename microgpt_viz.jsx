import { useState, useRef, useCallback, useEffect } from "react";

// ── Autograd engine (scalar) ──────────────────────────────────────
class V {
  constructor(data, children = [], localGrads = []) {
    this.data = data;
    this.grad = 0;
    this._children = children;
    this._localGrads = localGrads;
  }
  add(o) { o = o instanceof V ? o : new V(o); return new V(this.data + o.data, [this, o], [1, 1]); }
  mul(o) { o = o instanceof V ? o : new V(o); return new V(this.data * o.data, [this, o], [o.data, this.data]); }
  pow(n) { return new V(this.data ** n, [this], [n * this.data ** (n - 1)]); }
  log() { return new V(Math.log(this.data), [this], [1 / this.data]); }
  exp() { const e = Math.exp(this.data); return new V(e, [this], [e]); }
  relu() { return new V(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]); }
  neg() { return this.mul(-1); }
  sub(o) { o = o instanceof V ? o : new V(o); return this.add(o.neg()); }
  div(o) { o = o instanceof V ? o : new V(o); return this.mul(o.pow(-1)); }
  backward() {
    const topo = []; const visited = new Set();
    const build = (v) => { if (!visited.has(v)) { visited.add(v); for (const c of v._children) build(c); topo.push(v); } };
    build(this); this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v._children.length; j++) v._children[j].grad += v._localGrads[j] * v.grad;
    }
  }
}

// ── Helpers ───────────────────────────────────────────────────────
function multigauss(n, std) { const a = []; for (let i = 0; i < n; i++) { let u, v, s; do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0); const m = Math.sqrt(-2 * Math.log(s) / s); a.push(u * m * std); } return a; }
function matrix(r, c, std = 0.08) { const g = multigauss(r * c, std); return Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => new V(g[i * c + j]))); }
function linear(x, w) { return w.map(row => row.reduce((s, wi, i) => s.add(wi.mul(x[i])), new V(0))); }
function softmax(logits) { const mx = Math.max(...logits.map(v => v.data)); const exps = logits.map(v => v.sub(mx).exp()); const tot = exps.reduce((a, b) => a.add(b)); return exps.map(e => e.div(tot)); }
function rmsnorm(x) { let ms = x.reduce((s, xi) => s.add(xi.mul(xi)), new V(0)).div(x.length); const scale = ms.add(1e-5).pow(-0.5); return x.map(xi => xi.mul(scale)); }

// ── Names dataset (built-in) ─────────────────────────────────────
const NAMES = [
  "emma","olivia","ava","sophia","mia","isabella","charlotte","amelia","harper","evelyn",
  "luna","ella","elizabeth","sofia","emily","avery","scarlett","aria","penelope","chloe",
  "riley","nora","lily","eleanor","hazel","aurora","violet","stella","hannah","zoe",
  "liam","noah","oliver","james","elijah","william","henry","lucas","benjamin","jack",
  "leo","daniel","owen","samuel","mason","logan","jackson","aiden","oscar","ethan",
  "anna","kai","max","lia","ivy","ada","eva","leo","eli","ian"
];

// ── Build model ──────────────────────────────────────────────────
function buildModel() {
  const uchars = [...new Set(NAMES.join(""))].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  const nEmbd = 12, nHead = 3, headDim = 4, blockSize = 12, nLayer = 1;

  const sd = { wte: matrix(vocabSize, nEmbd), wpe: matrix(blockSize, nEmbd), lm_head: matrix(vocabSize, nEmbd) };
  for (let i = 0; i < nLayer; i++) {
    sd[`l${i}.wq`] = matrix(nEmbd, nEmbd); sd[`l${i}.wk`] = matrix(nEmbd, nEmbd);
    sd[`l${i}.wv`] = matrix(nEmbd, nEmbd); sd[`l${i}.wo`] = matrix(nEmbd, nEmbd);
    sd[`l${i}.fc1`] = matrix(4 * nEmbd, nEmbd); sd[`l${i}.fc2`] = matrix(nEmbd, 4 * nEmbd);
  }
  const params = []; for (const m of Object.values(sd)) for (const row of m) for (const p of row) params.push(p);

  return { uchars, BOS, vocabSize, nEmbd, nHead, headDim, blockSize, nLayer, sd, params };
}

function gpt(tokenId, posId, keys, values, model) {
  const { sd, nLayer, nHead, headDim } = model;
  let x = sd.wte[tokenId].map((t, i) => t.add(sd.wpe[posId][i]));
  x = rmsnorm(x);
  for (let li = 0; li < nLayer; li++) {
    const xr = x; x = rmsnorm(x);
    const q = linear(x, sd[`l${li}.wq`]), k = linear(x, sd[`l${li}.wk`]), v = linear(x, sd[`l${li}.wv`]);
    keys[li].push(k); values[li].push(v);
    const xAttn = [];
    for (let h = 0; h < nHead; h++) {
      const hs = h * headDim;
      const qh = q.slice(hs, hs + headDim);
      const kh = keys[li].map(ki => ki.slice(hs, hs + headDim));
      const vh = values[li].map(vi => vi.slice(hs, hs + headDim));
      const al = kh.map(kt => qh.reduce((s, qi, j) => s.add(qi.mul(kt[j])), new V(0)).div(Math.sqrt(headDim)));
      const aw = softmax(al);
      for (let j = 0; j < headDim; j++) xAttn.push(aw.reduce((s, w, t) => s.add(w.mul(vh[t][j])), new V(0)));
    }
    x = linear(xAttn, sd[`l${li}.wo`]).map((v, i) => v.add(xr[i]));
    const xr2 = x; x = rmsnorm(x);
    x = linear(x, sd[`l${li}.fc1`]).map(v => v.relu());
    x = linear(x, sd[`l${li}.fc2`]).map((v, i) => v.add(xr2[i]));
  }
  return linear(x, sd.lm_head);
}

// ── Weight matrix snapshot ───────────────────────────────────────
function snapshotMatrix(mat) { return mat.map(row => row.map(v => v.data)); }

// ── Heatmap canvas renderer ──────────────────────────────────────
function drawHeatmap(canvas, data, label) {
  if (!canvas) return;
  const rows = data.length, cols = data[0].length;
  const maxDim = 200;
  const cellW = Math.max(1, Math.min(8, Math.floor(maxDim / cols)));
  const cellH = Math.max(1, Math.min(8, Math.floor(maxDim / rows)));
  const w = cols * cellW, h = rows * cellH;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  let mn = Infinity, mx = -Infinity;
  for (const row of data) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const range = mx - mn || 1;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const t = (data[r][c] - mn) / range;
    // blue -> white -> red
    const red = t > 0.5 ? 255 : Math.round(255 * t * 2);
    const blue = t < 0.5 ? 255 : Math.round(255 * (1 - t) * 2);
    const green = t > 0.5 ? Math.round(255 * (1 - t) * 2) : Math.round(255 * t * 2);
    ctx.fillStyle = `rgb(${red},${green},${blue})`;
    ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
  }
}

// ── Component ────────────────────────────────────────────────────
function MatrixCard({ label, dims, canvasRef }) {
  return (
    <div style={{ background: "#1a1a2e", borderRadius: 8, padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#e0e0ff", fontWeight: 600 }}>{label}</span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#888" }}>{dims}</span>
      </div>
      <canvas ref={canvasRef} style={{ width: "100%", borderRadius: 4, imageRendering: "pixelated" }} />
    </div>
  );
}

export default function MicroGPTViz() {
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(null);
  const [running, setRunning] = useState(false);
  const [samples, setSamples] = useState([]);
  const [speed, setSpeed] = useState(1);
  const modelRef = useRef(null);
  const adamRef = useRef(null);
  const canvasRefs = useRef({});
  const totalSteps = 500;

  const matrixKeys = ["wte", "wpe", "lm_head", "l0.wq", "l0.wk", "l0.wv", "l0.wo", "l0.fc1", "l0.fc2"];
  const getRef = (k) => { if (!canvasRefs.current[k]) canvasRefs.current[k] = { current: null }; return canvasRefs.current[k]; };

  const init = useCallback(() => {
    const model = buildModel();
    modelRef.current = model;
    adamRef.current = { m: new Float64Array(model.params.length), v: new Float64Array(model.params.length) };
    setStep(0); setLoss(null); setSamples([]);
    requestAnimationFrame(() => {
      for (const k of matrixKeys) drawHeatmap(getRef(k).current, snapshotMatrix(model.sd[k]), k);
    });
  }, []);

  useEffect(() => { init(); }, [init]);

  const trainStep = useCallback((currentStep) => {
    const model = modelRef.current;
    const adam = adamRef.current;
    const { uchars, BOS, blockSize, nLayer, params } = model;
    const doc = NAMES[currentStep % NAMES.length];
    const tokens = [BOS, ...doc.split("").map(c => uchars.indexOf(c)), BOS];
    const n = Math.min(blockSize, tokens.length - 1);
    const keys = Array.from({ length: nLayer }, () => []);
    const values = Array.from({ length: nLayer }, () => []);
    let losses = [];
    for (let pos = 0; pos < n; pos++) {
      const logits = gpt(tokens[pos], pos, keys, values, model);
      const probs = softmax(logits);
      losses.push(probs[tokens[pos + 1]].log().neg());
    }
    let lossVal = losses.reduce((a, b) => a.add(b), new V(0)).div(n);
    lossVal.backward();
    const lr = 0.01 * (1 - currentStep / totalSteps);
    for (let i = 0; i < params.length; i++) {
      adam.m[i] = 0.85 * adam.m[i] + 0.15 * params[i].grad;
      adam.v[i] = 0.99 * adam.v[i] + 0.01 * params[i].grad ** 2;
      const mh = adam.m[i] / (1 - 0.85 ** (currentStep + 1));
      const vh = adam.v[i] / (1 - 0.99 ** (currentStep + 1));
      params[i].data -= lr * mh / (Math.sqrt(vh) + 1e-8);
      params[i].grad = 0;
    }
    return lossVal.data;
  }, []);

  const generate = useCallback(() => {
    const model = modelRef.current;
    const { uchars, BOS, vocabSize, blockSize, nLayer } = model;
    const out = [];
    for (let s = 0; s < 10; s++) {
      const keys = Array.from({ length: nLayer }, () => []);
      const values = Array.from({ length: nLayer }, () => []);
      let tid = BOS; const chars = [];
      for (let pos = 0; pos < blockSize; pos++) {
        const logits = gpt(tid, pos, keys, values, model);
        const probs = softmax(logits.map(l => l.div(0.5)));
        const weights = probs.map(p => p.data);
        let r = Math.random(), cum = 0;
        tid = 0;
        for (let i = 0; i < weights.length; i++) { cum += weights[i]; if (r < cum) { tid = i; break; } }
        if (tid === BOS) break;
        chars.push(uchars[tid]);
      }
      out.push(chars.join(""));
    }
    setSamples(out);
  }, []);

  const runLoop = useCallback(() => {
    if (!modelRef.current) return;
    setRunning(true);
    let s = 0;
    const tick = () => {
      setStep(prev => {
        if (prev >= totalSteps) { setRunning(false); generate(); return prev; }
        const batchSize = speed;
        let l = 0;
        for (let i = 0; i < batchSize && prev + i < totalSteps; i++) {
          l = trainStep(prev + i);
          s = prev + i + 1;
        }
        setLoss(l);
        if (s % Math.max(1, Math.floor(10 / speed)) === 0 || s >= totalSteps) {
          for (const k of matrixKeys) drawHeatmap(getRef(k).current, snapshotMatrix(modelRef.current.sd[k]), k);
        }
        if (s >= totalSteps) { setRunning(false); generate(); }
        else setTimeout(tick, 0);
        return s;
      });
    };
    tick();
  }, [speed, trainStep, generate]);

  const pct = (step / totalSteps) * 100;

  return (
    <div style={{ background: "#0f0f1a", color: "#e0e0ff", minHeight: "100vh", fontFamily: "'JetBrains Mono', 'Fira Code', monospace", padding: 20 }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 4px", letterSpacing: 1 }}>microGPT</h1>
        <p style={{ fontSize: 11, color: "#666", margin: "0 0 16px" }}>live weight matrix visualization / {modelRef.current?.params.length ?? 0} parameters</p>

        {/* Controls */}
        <div style={{ display: "flex", gap: 10, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
          <button onClick={running ? null : runLoop} disabled={running || step >= totalSteps}
            style={{ background: running ? "#333" : "#4a47a3", color: "#fff", border: "none", borderRadius: 6, padding: "8px 18px", cursor: running ? "default" : "pointer", fontFamily: "inherit", fontSize: 12, fontWeight: 600 }}>
            {running ? "Training..." : step >= totalSteps ? "Done" : "Train"}
          </button>
          <button onClick={() => { init(); }} disabled={running}
            style={{ background: "#2a2a3e", color: "#aaa", border: "1px solid #333", borderRadius: 6, padding: "8px 14px", cursor: "pointer", fontFamily: "inherit", fontSize: 12 }}>
            Reset
          </button>
          <label style={{ fontSize: 11, color: "#888", display: "flex", alignItems: "center", gap: 6 }}>
            Speed
            <select value={speed} onChange={e => setSpeed(Number(e.target.value))} style={{ background: "#1a1a2e", color: "#ccc", border: "1px solid #333", borderRadius: 4, padding: "4px 8px", fontFamily: "inherit", fontSize: 11 }}>
              <option value={1}>1x</option><option value={3}>3x</option><option value={5}>5x</option><option value={10}>10x</option>
            </select>
          </label>
          {step >= totalSteps && <button onClick={generate} style={{ background: "#2e5944", color: "#afc", border: "none", borderRadius: 6, padding: "8px 14px", cursor: "pointer", fontFamily: "inherit", fontSize: 12 }}>Generate</button>}
        </div>

        {/* Progress */}
        <div style={{ display: "flex", gap: 20, alignItems: "center", marginBottom: 16, fontSize: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ background: "#1a1a2e", borderRadius: 4, height: 8, overflow: "hidden" }}>
              <div style={{ background: "linear-gradient(90deg, #4a47a3, #7b6cf6)", width: `${pct}%`, height: "100%", transition: "width 0.1s" }} />
            </div>
          </div>
          <span style={{ color: "#888", minWidth: 100 }}>step {step}/{totalSteps}</span>
          {loss !== null && <span style={{ color: loss < 2.5 ? "#6fdc8c" : "#f5a623" }}>loss: {loss.toFixed(4)}</span>}
        </div>

        {/* Weight matrices */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 10, marginBottom: 20 }}>
          {matrixKeys.map(k => {
            const m = modelRef.current?.sd[k];
            const dims = m ? `${m.length}x${m[0].length}` : "";
            return <MatrixCard key={k} label={k} dims={dims} canvasRef={el => { getRef(k).current = el; }} />;
          })}
        </div>

        {/* Samples */}
        {samples.length > 0 && (
          <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 14 }}>
            <div style={{ fontSize: 11, color: "#888", marginBottom: 8 }}>Generated names</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {samples.map((s, i) => (
                <span key={i} style={{ background: "#2a2a4e", padding: "4px 10px", borderRadius: 4, fontSize: 13, color: "#c8c8ff" }}>{s}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
