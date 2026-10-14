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
  // TODO(you): g_logits = g_model.forward(ids_view, &g_acts)
  (void)ids;
  (void)n;
  return -1;
}

const float* llm_logits() { return g_logits.data.data(); }
int llm_vocab_size() { return g_model.cfg.vocab; }
int llm_n_layers() { return g_model.cfg.n_layer; }

// part 0 = attention weights for the last position [n_head, ctx]
// part 1 = mlp activations [4d]
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
