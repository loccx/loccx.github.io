import type { Interface } from "../ui/manager";

/**
 * llm/ — a tiny transformer trained only on corpus.txt, running as wasm.
 * JS does tokenization + sampling; the C++ side does every matmul.
 * If ../public/llm/ isn't built yet this renders the build instructions.
 *
 * The canvas viz reads activations straight out of wasm memory each step:
 * attention heatmaps per head, every neuron's activation per stage, and
 * the next-token distribution. See llm/README.md §5 for the contract.
 */

interface LlmModule {
  _llm_load(w: number, len: number): number;
  _llm_forward(ids: number, n: number): number;
  _llm_logits(): number;
  _llm_embedding(): number;
  _llm_vocab_size(): number;
  _llm_ctx(): number;
  _llm_d_model(): number;
  _llm_n_layers(): number;
  _llm_n_head(): number;
  _llm_activations(layer: number, part: number): number;
  _llm_activations_len(layer: number, part: number): number;
  _malloc(n: number): number;
  _free(p: number): void;
  HEAPF32: Float32Array;
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
}

const LLM_JS_URL = "/llm/llm.js";
const STEP_MS = 110;
const FONT = "11px 'BigBlue Terminal', monospace";
const INK = "#b8b8b8";
const DIM = "#555";
const POS = "95,135,255"; // #5f87ff
const NEG = "135,95,215"; // #875fd7

const glyph = (c: string) => (c === "\n" ? "¶" : c === " " ? "·" : c);

export class LlmInterface implements Interface {
  readonly id = "llm";
  private stopped = false;

  mount(root: HTMLElement) {
    this.stopped = false;
    const wrap = document.createElement("div");
    wrap.className = "fixed inset-0 overflow-y-auto";
    wrap.innerHTML = `
      <article style="max-width: 72ch; padding: 1.5em 2ch">
        <h1>llm/</h1>
        <p class="muted">a tiny transformer that only knows me. every neuron below is real.</p>
        <p id="llm-status" class="muted">loading…</p>
        <div id="llm-ui" style="display:none">
          <p>
            Q: <input id="llm-prompt" style="background:#000;border:1px solid #333;color:#b8b8b8;padding:2px 6px;width:32ch"
                value="who are you" />
            <a href="#" id="llm-go">[generate]</a>
          </p>
          <pre id="llm-out" style="min-height:4em"></pre>
          <canvas id="llm-viz" style="width:100%"></canvas>
        </div>
      </article>`;
    root.appendChild(wrap);

    const status = wrap.querySelector<HTMLElement>("#llm-status")!;
    const ui = wrap.querySelector<HTMLElement>("#llm-ui")!;

    this.boot(status, ui, wrap).catch(() => {
      status.textContent =
        "model not trained yet — see llm/README.md (make, then make wasm)";
    });
  }

  unmount() {
    this.stopped = true;
  }

  private async boot(status: HTMLElement, ui: HTMLElement, wrap: HTMLElement) {
    const [vocabText, weights, mod] = await Promise.all([
      fetch("/llm/vocab.txt").then((r) => {
        if (!r.ok) throw new Error("vocab");
        return r.text();
      }),
      fetch("/llm/weights.bin").then((r) => {
        if (!r.ok) throw new Error("weights");
        return r.arrayBuffer();
      }),
      import(/* @vite-ignore */ LLM_JS_URL).then((m) =>
        (m.default as () => Promise<LlmModule>)(),
      ),
    ]);

    const itos = vocabText
      .split("\n")
      .filter((l) => l.length)
      .map((l) => (l === "\\n" ? "\n" : l));
    const stoi = new Map([...itos].map((c, i) => [c, i]));
    const encode = (s: string) => [...s].map((c) => stoi.get(c) ?? 0);

    const wptr = mod._malloc(weights.byteLength);
    mod.HEAPU8.set(new Uint8Array(weights), wptr);
    if (mod._llm_load(wptr, weights.byteLength) !== 0) {
      mod._free(wptr);
      throw new Error("load");
    }
    mod._free(wptr);

    status.textContent = `loaded · ${mod._llm_vocab_size()} tokens · ${mod._llm_n_layers()} layers · ${mod._llm_d_model()}d`;
    ui.style.display = "";

    const prompt = wrap.querySelector<HTMLInputElement>("#llm-prompt")!;
    const out = wrap.querySelector<HTMLElement>("#llm-out")!;
    const canvas = wrap.querySelector<HTMLCanvasElement>("#llm-viz")!;
    const maxCtx = mod._llm_ctx();
    const iptr = mod._malloc(maxCtx * 4);
    let busy = false;

    const run = async () => {
      if (busy) return;
      busy = true;
      const text = `Q: ${prompt.value}\nA:`;
      const ids = encode(text).slice(0, maxCtx);
      let generated = "";

      while (!this.stopped) {
        mod.HEAP32.set(ids, iptr / 4);
        if (mod._llm_forward(iptr, ids.length) !== 0) break;
        const v = mod._llm_vocab_size();
        const lo = mod._llm_logits() / 4;
        const logits = mod.HEAPF32.subarray(lo, lo + v);
        let best = 0;
        for (let i = 1; i < v; i++) if (logits[i] > logits[best]) best = i;
        const ch = itos[best];
        generated += ch;
        ids.push(best);
        out.textContent = text + generated;
        drawViz(canvas, mod, ids, itos);
        if (ch === "\n" && generated.endsWith("\n\n")) break;
        if (ids.length >= maxCtx || generated.length >= 200) break;
        await new Promise((r) => setTimeout(r, STEP_MS));
      }
      busy = false;
    };

    wrap.querySelector("#llm-go")!.addEventListener("click", (e) => {
      e.preventDefault();
      run();
    });
  }
}

/* ------------------------------------------------------------------ */
/* viz — everything below draws one frame for the current forward pass */

function drawViz(
  canvas: HTMLCanvasElement,
  mod: LlmModule,
  ids: number[],
  itos: string[],
) {
  const n = ids.length;
  const L = mod._llm_n_layers();
  const H = mod._llm_n_head();
  const d = mod._llm_d_model();
  const C = mod._llm_ctx();
  const w = canvas.clientWidth || 620;
  const dpr = window.devicePixelRatio || 1;

  // cell size shared by the attention maps (2 maps per row)
  const cs = Math.max(2, Math.min(9, Math.floor((w / 2 - 48) / n)));
  const mapPx = n * cs;
  const seqRows = Math.ceil((n * 11) / (w - 16));

  const height =
    20 + seqRows * 16 + 8 + // sequence
    20 + Math.ceil((L * H) / 2) * (mapPx + 30) + // attention
    20 + (1 + L * 2) * 16 + // residual stream
    20 + L * 14 + // mlp hidden
    20 + 8 * 15 + // next token
    16;
  canvas.width = w * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;

  const g = canvas.getContext("2d")!;
  g.scale(dpr, dpr);
  g.font = FONT;
  g.fillStyle = "#000";
  g.fillRect(0, 0, w, height);
  g.textBaseline = "top";

  let y = 0;
  const label = (t: string) => {
    y += 6;
    g.fillStyle = DIM;
    g.fillText(t, 0, y);
    y += 14;
  };

  // -- sequence: every token, last one inverted --------------------
  label("sequence");
  let x = 0;
  for (let i = 0; i < n; i++) {
    if (x + 11 > w) {
      x = 0;
      y += 16;
    }
    if (i === n - 1) {
      g.fillStyle = INK;
      g.fillRect(x, y, 10, 15);
      g.fillStyle = "#000";
    } else {
      g.fillStyle = INK;
    }
    g.fillText(glyph(itos[ids[i]]), x + 1, y + 2);
    x += 11;
  }
  y += 24;

  // -- attention: n_head maps per layer, weight = brightness -------
  label("attention");
  const slotW = w / 2;
  for (let l = 0; l < L; l++) {
    const ptr = mod._llm_activations(l, 0);
    if (!ptr) continue;
    const attn = mod.HEAPF32.subarray(ptr / 4, ptr / 4 + H * C * C);
    for (let h = 0; h < H; h++) {
      const slot = l * H + h;
      const ox = (slot % 2) * slotW;
      const oy = y + Math.floor(slot / 2) * (mapPx + 30);
      g.fillStyle = DIM;
      g.fillText(`l${l} h${h}`, ox, oy);
      const base = h * C * C;
      for (let q = 0; q < n; q++) {
        for (let k = 0; k <= q; k++) {
          const a = attn[base + q * C + k];
          if (a <= 0) continue;
          g.fillStyle = `rgba(${POS},${Math.pow(a, 0.6).toFixed(3)})`;
          g.fillRect(ox + k * cs, oy + 14 + q * cs, cs, cs);
        }
      }
      // outline the live row — what the current token is looking at
      g.fillStyle = `rgba(${POS},0.5)`;
      g.fillRect(ox, oy + 14 + (n - 1) * cs, mapPx, 1);
    }
  }
  y += Math.ceil((L * H) / 2) * (mapPx + 30);

  // -- residual stream: every neuron, one row per stage ------------
  label("residual stream");
  const stage = (name: string, ptr: number, len: number) => {
    g.fillStyle = DIM;
    g.fillText(name, 0, y + 2);
    if (ptr) {
      const acts = mod.HEAPF32.subarray(ptr / 4, ptr / 4 + len);
      let peak = 1e-9;
      for (const a of acts) peak = Math.max(peak, Math.abs(a));
      const cw = Math.min(14, (w - 96) / len);
      for (let i = 0; i < len; i++) {
        const a = acts[i];
        const t = Math.min(Math.abs(a) / peak, 1);
        g.fillStyle = `rgba(${a >= 0 ? POS : NEG},${(0.08 + 0.92 * t).toFixed(3)})`;
        g.fillRect(96 + i * cw, y + 2, cw - 1, 12);
      }
    }
    y += 16;
  };

  stage("emb", mod._llm_embedding(), d);
  for (let l = 0; l < L; l++) {
    stage(`l${l} attn`, mod._llm_activations(l, 1), d);
    stage(`l${l} mlp`, mod._llm_activations(l, 3), d);
  }

  // -- mlp hidden: 4d neurons per layer -----------------------------
  label("mlp hidden");
  for (let l = 0; l < L; l++) {
    const len = mod._llm_activations_len(l, 2);
    const ptr = mod._llm_activations(l, 2);
    if (!ptr || !len) continue;
    const acts = mod.HEAPF32.subarray(ptr / 4, ptr / 4 + len);
    const cw = (w - 40) / len;
    let peak = 1e-9;
    for (const a of acts) peak = Math.max(peak, Math.abs(a));
    for (let i = 0; i < len; i++) {
      const a = acts[i];
      const t = Math.min(Math.abs(a) / peak, 1);
      g.fillStyle = `rgba(${a >= 0 ? POS : NEG},${(0.08 + 0.92 * t).toFixed(3)})`;
      g.fillRect(40 + i * cw, y, cw - 0.5, 11);
    }
    g.fillStyle = DIM;
    g.fillText(`l${l}`, 0, y + 1);
    y += 14;
  }

  // -- next token: softmax over logits -------------------------------
  label("next token");
  const v = mod._llm_vocab_size();
  const lo = mod._llm_logits() / 4;
  const logits = mod.HEAPF32.subarray(lo, lo + v);
  let mx = -Infinity;
  for (const z of logits) mx = Math.max(mx, z);
  let sum = 0;
  const probs = [...logits].map((z) => (sum += Math.exp(z - mx), Math.exp(z - mx)));
  const top = probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 8);
  for (const { p, i } of top) {
    g.fillStyle = INK;
    g.fillText(`'${glyph(itos[i])}'`, 0, y + 1);
    g.fillStyle = `rgba(${POS},0.85)`;
    g.fillRect(32, y + 3, (w - 140) * p, 10);
    g.fillStyle = DIM;
    g.fillText(p.toFixed(3), 40 + (w - 140) * p, y + 1);
    y += 15;
  }
}
