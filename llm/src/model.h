#pragma once

#include "tensor.h"
#include <string>
#include <vector>

struct Config {
  int vocab = 0;
  int ctx = 64;     // max sequence length
  int d_model = 32;
  int n_layer = 2;
  int n_head = 2;
};

struct LayerParams {
  Tensor ln1_w, ln1_b;
  Tensor wq, wk, wv, wo; // each [d, d]
  Tensor ln2_w, ln2_b;
  Tensor fc1, b1;        // [d, 4d], [4d]
  Tensor fc2, b2;        // [4d, d], [d]
};

struct Model {
  Config cfg;
  Tensor token_emb;            // [vocab, d]
  Tensor pos_emb;              // [ctx, d]
  std::vector<LayerParams> layers;
  Tensor lnf_w, lnf_b;         // final layernorm
  // unembedding is token_emb (tied)

  // TODO(you): kaiming-ish init — small random values, e.g. N(0, 0.02)
  void init(int vocab_size);

  // TODO(you): forward over the full sequence ids (len <= ctx).
  // returns logits [n, vocab]. fills `activations` if non-null:
  //   [0]              = last-position embedding (token_emb + pos_emb) [d]
  //   [1 + l*4 + 0]    = attention weights [n_head, ctx, ctx]
  //   [1 + l*4 + 1]    = attn block out at last position [d]
  //   [1 + l*4 + 2]    = mlp hidden at last position [4d]
  //   [1 + l*4 + 3]    = mlp block out at last position [d]
  // (the site's viz reads exactly these — see README §5)
  Tensor forward(const std::vector<int>& ids,
                 std::vector<Tensor>* activations = nullptr);

  // TODO(you): serialize every tensor to weights.bin in the README order
  bool save(const std::string& path) const;
  bool load(const std::string& path);
};
