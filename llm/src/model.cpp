#include "model.h"

#include <cstdio>
#include <cstring>
#include <fstream>

// TODO(you): this is the whole model. see README "2. model.cpp" for the
// exact forward equations and "4. export" for the weights.bin layout.

void Model::init(int vocab_size) {
  cfg.vocab = vocab_size;
  // allocate every tensor to its shape, then fill with N(0, 0.02)
}

Tensor Model::forward(const std::vector<int>& ids,
                      std::vector<Tensor>* activations) {
  // token + position embedding
  // n_layer x { attention block, mlp block } with residuals + layernorms
  // final layernorm, then logits = x * token_emb^T
  Tensor logits{{(int)ids.size(), cfg.vocab}};
  logits.data.assign((size_t)ids.size() * cfg.vocab, 0.f);
  return logits;
}

bool Model::save(const std::string& path) const {
  // magic "KEYGPT01", i32 header, then raw f32s in README order
  return false;
}

bool Model::load(const std::string& path) {
  return false;
}
