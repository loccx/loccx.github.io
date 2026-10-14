import type { Interface } from "../ui/manager";

/**
 * llm/ — a tiny transformer trained only on corpus.txt, running as wasm.
 * JS does tokenization + sampling; the C++ side does every matmul.
 * If ../public/llm/ isn't built yet this renders the build instructions.
 */

interface LlmModule {
  _llm_load(w: number, len: number): number;
  _llm_forward(ids: number, n: number): number;
  _llm_logits(): number;
  _llm_vocab_size(): number;
  _llm_n_layers(): number;
  _llm_activations(layer: number, part: number): number;
  _llm_activations_len(layer: number, part: number): number;
  _malloc(n: number): number;
  _free(p: number): void;
  HEAPF32: Float32Array;
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
}

const LLM_JS_URL = "/llm/llm.js";

export class LlmInterface implements Interface {
  readonly id = "llm";

  mount(root: HTMLElement) {
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
          <div id="llm-viz"></div>
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
    const encode = (s: string) =>
      [...s].map((c) => stoi.get(c) ?? 0);

    const wptr = mod._malloc(weights.byteLength);
    mod.HEAPU8.set(new Uint8Array(weights), wptr);
    if (mod._llm_load(wptr, weights.byteLength) !== 0) {
      mod._free(wptr);
      throw new Error("load");
    }
    mod._free(wptr);

    status.textContent = `loaded · ${mod._llm_vocab_size()} tokens · ${mod._llm_n_layers()} layers`;
    ui.style.display = "";

    const prompt = wrap.querySelector<HTMLInputElement>("#llm-prompt")!;
    const out = wrap.querySelector<HTMLElement>("#llm-out")!;
    const viz = wrap.querySelector<HTMLElement>("#llm-viz")!;

    const run = () => {
      const text = `Q: ${prompt.value}\nA:`;
      const ids = encode(text);
      const iptr = mod._malloc(ids.length * 4);

      let generated = "";
      const pushToken = () => {
        mod.HEAP32.set(ids, iptr / 4);
        mod._llm_forward(iptr, ids.length);
        const v = mod._llm_vocab_size();
        const logits = mod.HEAPF32.subarray(
          mod._llm_logits() / 4,
          mod._llm_logits() / 4 + v,
        );
        let best = 0;
        for (let i = 1; i < v; i++) if (logits[i] > logits[best]) best = i;
        const ch = itos[best];
        generated += ch;
        ids.push(best);
        out.textContent = text + generated;
        drawViz(viz, mod);
        return ch === "\n" && generated.endsWith("\n\n");
      };

      while (!pushToken() && generated.length < 200);
      mod._free(iptr);
    };

    wrap.querySelector("#llm-go")!.addEventListener("click", (e) => {
      e.preventDefault();
      run();
    });
  }
}

function drawViz(viz: HTMLElement, mod: LlmModule) {
  viz.innerHTML = "";
  const layers = mod._llm_n_layers();
  for (let l = 0; l < layers; l++) {
    const n = mod._llm_activations_len(l, 1);
    if (!n) continue;
    const ptr = mod._llm_activations(l, 1) / 4;
    const acts = mod.HEAPF32.subarray(ptr, ptr + n);
    const row = document.createElement("div");
    row.innerHTML = `<span class="muted">layer ${l} mlp:</span> ` +
      [...acts]
        .map((a) => {
          const v = Math.min(Math.abs(a) * 40, 200) | 0;
          return `<span style="color:rgb(${v},${v},${v})">█</span>`;
        })
        .join("");
    viz.appendChild(row);
  }
}
