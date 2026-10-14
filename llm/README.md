# llm/

a tiny transformer, trained only on `corpus.txt`, that answers questions
about key. everything below is yours to implement — the plumbing (tokenizer,
build, wasm boundary, site viz) is already wired.

the point: a model small enough that you can see every wire. d_model=32,
2 layers, 2 heads, ~60k params. it memorizes the corpus; that's the feature.

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

## implement in this order

### 1. `src/tensor.h` — the math primitives
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
- sample a random window of ctx tokens from the corpus
- forward → cross-entropy loss on predicting token t+1 from t
- backprop. write it by hand (chain rule through each op) or do a
  micrograd-style autograd on the tensor — both teach you the same thing
- adam (m, v, beta1=.9, beta2=.999, eps=1e-8), lr ~1e-3
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
const float* llm_logits();                          // [vocab]
const float* llm_activations(int layer, int part);  // part 0=attn,1=mlp
int          llm_activations_len(int layer, int part);
```
js encodes text via vocab.txt, calls forward, samples logits itself, and
reads the activation pointers straight out of wasm memory for the viz.

## debugging

- loss not dropping after ~500 iters: check the causal mask, then check
  that softmax subtracts max, then check layernorm eps.
- model outputs one char forever: probably forgot `x +=` (residual) or
  pos_emb.
- js-side garble that trains fine natively: byte order / layout mismatch
  in weights.bin — dump the first few floats natively and compare.
