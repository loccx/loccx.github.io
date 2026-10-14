# llm/

a tiny transformer, trained only on `corpus.txt`, that answers questions
about key — running in the browser as webassembly, visualized down to
every neuron.

the point: a model small enough that you can see every wire. d_model=32,
2 layers, 2 heads, ~60k params. it memorizes the corpus; that's the feature.
everything is plain c++ with no dependencies — the sections below explain
each piece in the order you'd build them yourself.

## build

```sh
make            # ./train — native, needs only g++
make wasm       # ../public/llm/llm.js + .wasm — needs emscripten
```

emscripten (one-time):
```sh
git clone https://github.com/emscripten-core/emsdk ~/emsdk
~/emsdk/emsdk install latest && ~/emsdk/emsdk activate latest
source ~/emsdk/emsdk_env.sh   # add to .bashrc
```

## corpus

`corpus.txt` is the entire world the model knows. format:

```
Q: where do you go to school
A: ucsb

Q: what do you build
A: c++ stuff, small neural nets, keyboards
```

inference convention: feed `Q: <question>\nA:` then generate until `\n\n`.
write lots of pairs. rephrase the same fact a few ways — that's your dataset.

## how it works (read in this order)

### 1. `src/tensor.cpp` — the math primitives
- `matmul(a, b)` — [m,k] x [k,n]
- `softmax(v)` — subtract max first or it explodes
- `layernorm(v, w, b)` — mean/var + affine, eps=1e-5
- `gelu(v)` — tanh approx is fine

### 2. `src/model.cpp` — the forward pass
per token position t:
```
x = token_emb[tok[t]] + pos_emb[t]          // [d]
for each layer:
  a  = layernorm(x)
  q,k,v = a*wq, a*wk, a*wv                  // [d] each
  for each head h: attn = softmax(q_h·k_h / sqrt(dh)) masked to t'<t
                   x += concat_h(attn·v_h) * wo
  x += mlp(layernorm(x))                    // fc1 -> gelu -> fc2
logits = layernorm(x) * token_emb^T         // tied weights [vocab]
```
causal mask: position t can only attend to keys from t' <= t. this is the
one bug everyone writes — a model that reads the future looks amazing in
training and generates garbage.

### 3. `src/train.cpp` — the learning loop
- a ~40-line tensor autograd: each op records a backward closure, the
  tape walks in reverse. (bug worth knowing: closures that capture their
  own node create a shared_ptr cycle — pass the node as an argument)
- sample a random ctx window → forward → cross-entropy → backward → adam
  (m, v, beta1=.9, beta2=.999, eps=1e-8), lr 1e-3, 60k iters, ~10 min on cpu
- overfit. corpus is kilobytes; memorization is the goal, not a bug

### 4. `src/export` — weights.bin + vocab.txt
binary, little-endian f32. order:
```
magic "KEYGPT01"
i32 vocab, ctx, d_model, n_layer, n_head
f32 token_emb[vocab][d], pos_emb[ctx][d]
per layer: ln1_w,ln1_b, wq,wk,wv,wo [d][d] each,
           ln2_w,ln2_b, fc1[d][4d],b1[4d], fc2[4d][d],b2[d]
f32 lnf_w, lnf_b
```
vocab.txt: one char per line, line number = token id.

### 5. `src/wasm.cpp` — the browser boundary (stubs exist)
contract the site expects (already coded in src/interfaces/llm.ts):
```c
int          llm_load(const uint8_t* weights, int len);
int          llm_forward(const int* ids, int n);    // returns 0 ok
const float* llm_logits();                          // [vocab] last pos
const float* llm_embedding();                       // [d]    last pos
int          llm_vocab_size();
int          llm_ctx();                             // max seq len
int          llm_d_model();
int          llm_n_layers();
int          llm_n_head();
const float* llm_activations(int layer, int part);
int          llm_activations_len(int layer, int part);
```
`part` per layer: 0 = attention weights [n_head, ctx, ctx] indexed
`h*ctx*ctx + q*ctx + k` (only q,k < n meaningful), 1 = attn block out at
last position [d], 2 = mlp hidden at last position [4d], 3 = mlp block out
at last position [d]. suggested `activations` layout in Model::forward:
`[0]`=embedding then per layer the four part tensors, i.e.
`acts[1 + layer*4 + part]`.

js encodes text via vocab.txt, calls forward, samples logits itself, and
reads the activation pointers straight out of wasm memory for the viz:
attention heatmaps per head, every neuron's activation per stage, and the
next-token distribution — so save what forward computes in g_acts.

## debugging

- loss not dropping after ~500 iters: check the causal mask, then check
  that softmax subtracts max, then check layernorm eps.
- model outputs one char forever: probably forgot `x +=` (residual) or
  pos_emb.
- js-side garble that trains fine natively: byte order / layout mismatch
  in weights.bin — dump the first few floats natively and compare.
