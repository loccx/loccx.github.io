#include "model.h"
#include "tokenizer.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ */
/* minimal tensor autograd — each op records a closure that scatters   */
/* the output's grad into its inputs. build the graph per iteration,   */
/* then walk it in reverse.                                            */

struct Node {
  Tensor v;                 // value
  Tensor g;                 // gradient, lazily allocated
  Tensor m, s;              // adam moments (params only)
  bool has_g = false;
  bool is_param = false;
  std::function<void(Node&)> bw; // propagate self.g into inputs
};
using V = std::shared_ptr<Node>;

static std::vector<V> tape;

static V wrap(const Tensor& t) {
  V n = std::make_shared<Node>();
  n->v = t;
  tape.push_back(n);
  return n;
}

static V param(const Tensor& t) {
  V n = std::make_shared<Node>();
  n->v = t;
  n->g = t;
  n->m = t;
  n->s = t;
  std::fill(n->g.data.begin(), n->g.data.end(), 0.f);
  std::fill(n->m.data.begin(), n->m.data.end(), 0.f);
  std::fill(n->s.data.begin(), n->s.data.end(), 0.f);
  n->has_g = true;
  n->is_param = true;
  return n;
}

static void acc(const V& n, size_t i, float dv) {
  if (!n->has_g) {
    n->g = n->v;
    std::fill(n->g.data.begin(), n->g.data.end(), 0.f);
    n->has_g = true;
  }
  n->g.data[i] += dv;
}

static void backward(const V& loss) {
  acc(loss, 0, 1.f);
  for (auto it = tape.rbegin(); it != tape.rend(); ++it)
    if ((*it)->bw) (*it)->bw(**it);
}

/* -- ops ------------------------------------------------------------ */

static V add_op(const V& a, const V& b) {
  V o = wrap(add(a->v, b->v));
  o->bw = [a, b](Node& self) {
    for (size_t i = 0; i < self.g.data.size(); i++) {
      acc(a, i, self.g.data[i]);
      acc(b, i, self.g.data[i]);
    }
  };
  return o;
}

// rowwise bias: x[n,d] + b[d]
static V add_bias(const V& x, const V& b) {
  Tensor y = x->v;
  const int n = y.shape[0], d = y.shape[1];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++) y.at(i, j) += b->v.data[j];
  V o = wrap(y);
  o->bw = [x, b, n, d](Node& self) {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < d; j++) {
        const float gy = self.g.at(i, j);
        acc(x, (size_t)i * d + j, gy);
        acc(b, (size_t)j, gy);
      }
  };
  return o;
}

static V matmul_op(const V& a, const V& b) {
  V o = wrap(matmul(a->v, b->v));
  const int m = a->v.shape[0], k = a->v.shape[1], n = b->v.shape[1];
  o->bw = [a, b, m, k, n](Node& self) {
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        const float gy = self.g.at(i, j);
        if (gy == 0.f) continue;
        for (int kk = 0; kk < k; kk++) {
          acc(a, (size_t)i * k + kk, gy * b->v.at(kk, j));
          acc(b, (size_t)kk * n + j, a->v.at(i, kk) * gy);
        }
      }
  };
  return o;
}

static V transpose_op(const V& a) {
  const int r = a->v.shape[0], c = a->v.shape[1];
  Tensor t{{c, r}};
  t.data.resize(a->v.data.size());
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++) t.data[(size_t)j * r + i] = a->v.at(i, j);
  V o = wrap(t);
  o->bw = [a, r, c](Node& self) {
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++)
        acc(a, (size_t)i * c + j, self.g.data[(size_t)j * r + i]);
  };
  return o;
}

static V scale_op(const V& a, float s) {
  Tensor y = a->v;
  for (float& v : y.data) v *= s;
  V o = wrap(y);
  o->bw = [a, s](Node& self) {
    for (size_t i = 0; i < self.g.data.size(); i++) acc(a, i, self.g.data[i] * s);
  };
  return o;
}

static V softmax_op(const V& x) {
  V o = wrap(softmax(x->v));
  const int k = x->v.shape.back();
  const int rows = (int)x->v.data.size() / k;
  o->bw = [x, k, rows](Node& self) {
    for (int r = 0; r < rows; r++) {
      const float* y = self.v.data.data() + (size_t)r * k;
      const float* gy = self.g.data.data() + (size_t)r * k;
      float dot = 0.f;
      for (int j = 0; j < k; j++) dot += gy[j] * y[j];
      for (int j = 0; j < k; j++) acc(x, (size_t)r * k + j, y[j] * (gy[j] - dot));
    }
  };
  return o;
}

static V layernorm_op(const V& x, const V& w, const V& b) {
  V o = wrap(layernorm(x->v, w->v, b->v));
  const int d = x->v.shape.back();
  const int rows = (int)x->v.data.size() / d;
  o->bw = [x, w, b, d, rows](Node& self) {
    for (int r = 0; r < rows; r++) {
      const float* xv = x->v.data.data() + (size_t)r * d;
      const float* gy = self.g.data.data() + (size_t)r * d;
      // recompute xhat + inv std
      float mean = 0.f;
      for (int j = 0; j < d; j++) mean += xv[j];
      mean /= d;
      float var = 0.f;
      for (int j = 0; j < d; j++) {
        const float c = xv[j] - mean;
        var += c * c;
      }
      const float inv = 1.f / std::sqrt(var / d + 1e-5f);
      std::vector<float> xh(d), dxh(d);
      float s1 = 0.f, s2 = 0.f;
      for (int j = 0; j < d; j++) {
        xh[j] = (xv[j] - mean) * inv;
        acc(w, (size_t)j, gy[j] * xh[j]);
        acc(b, (size_t)j, gy[j]);
        dxh[j] = gy[j] * w->v.data[j];
        s1 += dxh[j];
        s2 += dxh[j] * xh[j];
      }
      for (int j = 0; j < d; j++)
        acc(x, (size_t)r * d + j,
            (dxh[j] - s1 / d - xh[j] * s2 / d) * inv);
    }
  };
  return o;
}

static V gelu_op(const V& x) {
  V o = wrap(gelu(x->v));
  o->bw = [x](Node& self) {
    for (size_t i = 0; i < self.g.data.size(); i++) {
      const float v = x->v.data[i];
      const float u = 0.7978845608f * (v + 0.044715f * v * v * v);
      const float t = std::tanh(u);
      const float du = 0.7978845608f * (1.f + 3.f * 0.044715f * v * v);
      const float d = 0.5f * (1.f + t) + 0.5f * v * (1.f - t * t) * du;
      acc(x, i, self.g.data[i] * d);
    }
  };
  return o;
}

// columns c0..c1 of x [n,d]
static V slice_cols_op(const V& x, int c0, int c1) {
  const int n = x->v.shape[0], d = x->v.shape[1], w = c1 - c0;
  Tensor y{{n, w}};
  y.data.resize((size_t)n * w);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < w; j++) y.at(i, j) = x->v.at(i, c0 + j);
  V o = wrap(y);
  o->bw = [x, n, d, c0, w](Node& self) {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < w; j++)
        acc(x, (size_t)i * d + c0 + j, self.g.at(i, j));
  };
  return o;
}

static V concat_cols_op(const std::vector<V>& parts) {
  const int n = parts[0]->v.shape[0];
  int dtot = 0;
  for (auto& p : parts) dtot += p->v.shape[1];
  Tensor y{{n, dtot}};
  y.data.resize((size_t)n * dtot);
  int off = 0;
  for (auto& p : parts) {
    const int w = p->v.shape[1];
    for (int i = 0; i < n; i++)
      for (int j = 0; j < w; j++) y.at(i, off + j) = p->v.at(i, j);
    off += w;
  }
  V o = wrap(y);
  o->bw = [parts, n, dtot](Node& self) {
    int off = 0;
    for (auto& p : parts) {
      const int w = p->v.shape[1];
      for (int i = 0; i < n; i++)
        for (int j = 0; j < w; j++)
          acc(p, (size_t)i * w + j, self.g.at(i, off + j));
      off += w;
    }
  };
  return o;
}

// rows idx[i] of emb [v,d] -> out [n,d]; backward scatter-adds into emb
static V gather_rows(const V& emb, const std::vector<int>& idx) {
  const int n = (int)idx.size(), d = emb->v.shape[1];
  Tensor y{{n, d}};
  y.data.resize((size_t)n * d);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++) y.at(i, j) = emb->v.at(idx[i], j);
  V o = wrap(y);
  o->bw = [emb, idx, n, d](Node& self) {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < d; j++)
        acc(emb, (size_t)idx[i] * d + j, self.g.at(i, j));
  };
  return o;
}

// mean cross-entropy over positions: scalar node
static V ce_loss(const V& logits, const std::vector<int>& targets) {
  const int n = logits->v.shape[0], v = logits->v.shape[1];
  Tensor probs = softmax(logits->v);
  float loss = 0.f;
  for (int i = 0; i < n; i++) loss -= std::log(probs.at(i, targets[i]) + 1e-12f);
  loss /= n;
  Tensor scalar{{1}};
  scalar.data = {loss};
  V o = wrap(scalar);
  o->bw = [logits, probs, targets, n, v](Node& self) {
    (void)self;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < v; j++) {
        float gy = probs.at(i, j) - (j == targets[i] ? 1.f : 0.f);
        acc(logits, (size_t)i * v + j, gy / n);
      }
  };
  return o;
}

/* ------------------------------------------------------------------ */

static std::string read_file(const char* path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static void adam_step(V& p, float lr, float b1, float b2, float eps, int t) {
  const float bc1 = 1.f - std::pow(b1, (float)t);
  const float bc2 = 1.f - std::pow(b2, (float)t);
  for (size_t i = 0; i < p->v.data.size(); i++) {
    const float g = p->g.data[i];
    p->m.data[i] = b1 * p->m.data[i] + (1.f - b1) * g;
    p->s.data[i] = b2 * p->s.data[i] + (1.f - b2) * g * g;
    p->v.data[i] -= lr * (p->m.data[i] / bc1) / (std::sqrt(p->s.data[i] / bc2) + eps);
    p->g.data[i] = 0.f;
  }
}

int main() {
  const std::string corpus = read_file("corpus.txt");
  Tokenizer tok;
  tok.build(corpus);
  printf("vocab: %d chars, corpus: %zu bytes\n", tok.size(), corpus.size());
  std::vector<int> data = tok.encode(corpus);

  Model model;
  model.init(tok.size());
  const Config& cfg = model.cfg;
  const int d = cfg.d_model, H = cfg.n_head, dh = d / H, ctx = cfg.ctx;

  // every weight tensor as an autograd param; save-order mirrors weights.bin
  std::vector<V> params;
  auto mk = [&](Tensor& t) {
    V n = param(t);
    params.push_back(n);
    return n;
  };
  V wte = mk(model.token_emb), wpe = mk(model.pos_emb);
  struct LP { V ln1w, ln1b, wq, wk, wv, wo, ln2w, ln2b, fc1, b1, fc2, b2; };
  std::vector<LP> lp;
  for (auto& l : model.layers)
    lp.push_back({mk(l.ln1_w), mk(l.ln1_b), mk(l.wq), mk(l.wk), mk(l.wv),
                  mk(l.wo), mk(l.ln2_w), mk(l.ln2_b), mk(l.fc1), mk(l.b1),
                  mk(l.fc2), mk(l.b2)});
  V wlnf = mk(model.lnf_w), blnf = mk(model.lnf_b);

  // causal mask for a full ctx window: -inf above the diagonal
  Tensor mask{{ctx, ctx}};
  mask.data.assign((size_t)ctx * ctx, 0.f);
  for (int i = 0; i < ctx; i++)
    for (int j = i + 1; j < ctx; j++) mask.at(i, j) = -1e9f;

  std::vector<int> pos_idx(ctx);
  for (int i = 0; i < ctx; i++) pos_idx[i] = i;

  std::mt19937 rng(2024);
  const int iters = 60000;
  const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;

  for (int it = 1; it <= iters; it++) {
    tape.clear();
    const int s = (int)(rng() % (uint32_t)(data.size() - ctx - 1));
    std::vector<int> ids(data.begin() + s, data.begin() + s + ctx);
    std::vector<int> targets(data.begin() + s + 1, data.begin() + s + ctx + 1);

    V x = add_op(gather_rows(wte, ids), gather_rows(wpe, pos_idx));
    for (auto& l : lp) {
      V a = layernorm_op(x, l.ln1w, l.ln1b);
      V q = matmul_op(a, l.wq), k = matmul_op(a, l.wk), v = matmul_op(a, l.wv);
      std::vector<V> heads;
      for (int h = 0; h < H; h++) {
        V qh = slice_cols_op(q, h * dh, (h + 1) * dh);
        V kh = slice_cols_op(k, h * dh, (h + 1) * dh);
        V vh = slice_cols_op(v, h * dh, (h + 1) * dh);
        V sc = matmul_op(qh, transpose_op(kh));
        sc = add_op(scale_op(sc, 1.f / std::sqrt((float)dh)), wrap(mask));
        V probs = softmax_op(sc);
        heads.push_back(matmul_op(probs, vh));
      }
      V cat = concat_cols_op(heads);
      x = add_op(x, matmul_op(cat, l.wo));

      V m = layernorm_op(x, l.ln2w, l.ln2b);
      V hidden = gelu_op(add_bias(matmul_op(m, l.fc1), l.b1));
      x = add_op(x, add_bias(matmul_op(hidden, l.fc2), l.b2));
    }
    V xn = layernorm_op(x, wlnf, blnf);
    V logits = matmul_op(xn, transpose_op(wte));
    V loss = ce_loss(logits, targets);

    backward(loss);
    for (auto& p : params) adam_step(p, lr, b1, b2, eps, it);

    if (it % 200 == 0 || it == 1) printf("iter %5d  loss %.4f\n", it, loss->v.data[0]);
  }

  // copy trained params back into the model and export
  auto put = [](Tensor& dst, const V& p) { dst = p->v; };
  size_t pi = 0;
  put(model.token_emb, params[pi++]);
  put(model.pos_emb, params[pi++]);
  for (auto& l : model.layers)
    for (Tensor* t : {&l.ln1_w, &l.ln1_b, &l.wq, &l.wk, &l.wv, &l.wo,
                      &l.ln2_w, &l.ln2_b, &l.fc1, &l.b1, &l.fc2, &l.b2})
      put(*t, params[pi++]);
  put(model.lnf_w, params[pi++]);
  put(model.lnf_b, params[pi++]);

  if (!model.save("weights.bin")) printf("save failed\n");
  tok.save_vocab("vocab.txt");
  printf("saved weights.bin + vocab.txt\n");

  // sanity: greedy generation from a couple of prompts
  for (const char* prompt : {"Q: who are you\nA:", "Q: where do you work\nA:"}) {
    std::vector<int> ids = tok.encode(prompt);
    std::string gen;
    for (int i = 0; i < 80 && ids.size() < (size_t)ctx; i++) {
      Tensor logits = model.forward(ids);
      const float* row = logits.data.data() + (size_t)(ids.size() - 1) * cfg.vocab;
      int best = 0;
      for (int j = 1; j < cfg.vocab; j++) if (row[j] > row[best]) best = j;
      ids.push_back(best);
      char c = tok.itos[best];
      gen += c;
      if (c == '\n' && gen.size() >= 2 && gen[gen.size() - 2] == '\n') break;
    }
    printf("\n%s%s\n", prompt, gen.c_str());
  }
  return 0;
}
