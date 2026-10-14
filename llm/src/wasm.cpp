#include "model.h"

#include <cstdint>
#include <vector>

// wasm boundary — compiled by `make wasm` into ../public/llm/llm.js.
// the site's llm interface calls these directly; see README "5. wasm.cpp"
// for the contract. JS does tokenization + sampling; this file is math only.

static Model g_model;
static std::vector<Tensor> g_acts;
static Tensor g_logits;

extern "C" {

// parse weights.bin held in JS memory. returns 0 on success.
int llm_load(const uint8_t* weights, int len) {
  // TODO(you): same parse as Model::load but from a byte buffer
  (void)weights;
  (void)len;
  return -1;
}

// run forward over the token ids. fills logits + captured activations.
int llm_forward(const int* ids, int n) {
  // TODO(you): g_logits = last row of g_model.forward(ids_view, &g_acts)
  (void)ids;
  (void)n;
  return -1;
}

// [vocab] logits at the last position
const float* llm_logits() { return g_logits.data.data(); }

// [d] embedding of the last position (token_emb + pos_emb)
const float* llm_embedding() {
  // TODO(you): return g_acts[0]
  return nullptr;
}

int llm_vocab_size() { return g_model.cfg.vocab; }
int llm_ctx() { return g_model.cfg.ctx; }
int llm_d_model() { return g_model.cfg.d_model; }
int llm_n_layers() { return g_model.cfg.n_layer; }
int llm_n_head() { return g_model.cfg.n_head; }

// per-layer tensors the viz reads. parts:
//   0 = attention weights [n_head, ctx, ctx], index h*ctx*ctx + q*ctx + k
//       (only q,k < n meaningful; k > q is masked)
//   1 = attn block out at last position [d]  (post-residual)
//   2 = mlp hidden at last position [4d]
//   3 = mlp block out at last position [d]   (post-residual)
// suggested g_acts layout: [0]=embedding, then per layer 4 tensors in the
// part order above, so g_acts[1 + layer*4 + part].
const float* llm_activations(int layer, int part) {
  // TODO(you): index into g_acts
  (void)layer;
  (void)part;
  return nullptr;
}
int llm_activations_len(int layer, int part) {
  (void)layer;
  (void)part;
  return 0;
}

} // extern "C"
