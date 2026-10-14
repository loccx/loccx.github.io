#include "model.h"

#include <cstdint>
#include <vector>

// wasm boundary — compiled by `make wasm` into ../public/llm/llm.js.
// the site's llm interface calls these directly; see README §5 for the
// contract. JS does tokenization + sampling; this file is math only.

static Model g_model;
static std::vector<Tensor> g_acts;
static Tensor g_logits;

extern "C" {

int llm_load(const uint8_t* weights, int len) {
  return g_model.load_mem(weights, (size_t)len) ? 0 : -1;
}

// forward over all ids; keeps last-position logits + activations for viz
int llm_forward(const int* ids, int n) {
  if (n <= 0 || n > g_model.cfg.ctx) return -1;
  std::vector<int> v(ids, ids + n);
  Tensor logits = g_model.forward(v, &g_acts);
  const int vocab = g_model.cfg.vocab;
  g_logits = Tensor{{vocab}};
  g_logits.data.assign(logits.data.end() - vocab, logits.data.end());
  return 0;
}

const float* llm_logits() { return g_logits.data.data(); }
const float* llm_embedding() { return g_acts[0].data.data(); }

int llm_vocab_size() { return g_model.cfg.vocab; }
int llm_ctx() { return g_model.cfg.ctx; }
int llm_d_model() { return g_model.cfg.d_model; }
int llm_n_layers() { return g_model.cfg.n_layer; }
int llm_n_head() { return g_model.cfg.n_head; }

// parts: 0=attn weights [H,ctx,ctx], 1=attn out [d], 2=mlp hidden [4d],
//        3=mlp out [d] — all at the last position except part 0
const float* llm_activations(int layer, int part) {
  return g_acts[1 + layer * 4 + part].data.data();
}
int llm_activations_len(int layer, int part) {
  return (int)g_acts[1 + layer * 4 + part].data.size();
}

} // extern "C"
