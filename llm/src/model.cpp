#include "model.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

// the model: embed -> n_layer x { attn block, mlp block } -> lnf -> tied
// unembedding. forward also captures what the site's viz reads (the
// `activations` layout documented in model.h / README §5).

static Tensor zeros(std::vector<int> shape) {
  Tensor t{shape};
  size_t n = 1;
  for (int s : shape) n *= (size_t)s;
  t.data.assign(n, 0.f);
  return t;
}

static Tensor transpose2d(const Tensor& t) {
  Tensor out{{t.shape[1], t.shape[0]}};
  out.data.resize(t.data.size());
  for (int i = 0; i < t.shape[0]; i++)
    for (int j = 0; j < t.shape[1]; j++) out.data[(size_t)j * t.shape[0] + i] = t.at(i, j);
  return out;
}

// columns c0..c1 of x [n,d]
static Tensor slice_cols(const Tensor& x, int c0, int c1) {
  const int n = x.shape[0], w = c1 - c0;
  Tensor out{{n, w}};
  out.data.resize((size_t)n * w);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < w; j++) out.at(i, j) = x.at(i, c0 + j);
  return out;
}

static void add_row_bias(Tensor& x, const Tensor& b) {
  const int n = x.shape[0], d = x.shape[1];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++) x.at(i, j) += b.data[j];
}

static Tensor last_row(const Tensor& x) {
  const int n = x.shape[0], d = x.shape[1];
  Tensor out{{d}};
  out.data.assign(x.data.end() - d, x.data.end());
  (void)n;
  return out;
}

void Model::init(int vocab_size) {
  cfg.vocab = vocab_size;
  const int d = cfg.d_model;
  std::mt19937 rng(1337);
  std::normal_distribution<float> gauss(0.f, 0.02f);
  auto fill = [&](Tensor& t, std::vector<int> shape, bool random = true) {
    t = zeros(shape);
    if (random)
      for (float& v : t.data) v = gauss(rng);
  };

  fill(token_emb, {cfg.vocab, d});
  fill(pos_emb, {cfg.ctx, d});
  layers.resize(cfg.n_layer);
  for (auto& l : layers) {
    fill(l.ln1_w, {d}, false);
    fill(l.ln1_b, {d}, false);
    std::fill(l.ln1_w.data.begin(), l.ln1_w.data.end(), 1.f);
    fill(l.wq, {d, d});
    fill(l.wk, {d, d});
    fill(l.wv, {d, d});
    fill(l.wo, {d, d});
    fill(l.ln2_w, {d}, false);
    fill(l.ln2_b, {d}, false);
    std::fill(l.ln2_w.data.begin(), l.ln2_w.data.end(), 1.f);
    fill(l.fc1, {d, 4 * d});
    fill(l.b1, {4 * d}, false);
    fill(l.fc2, {4 * d, d});
    fill(l.b2, {d}, false);
  }
  fill(lnf_w, {d}, false);
  fill(lnf_b, {d}, false);
  std::fill(lnf_w.data.begin(), lnf_w.data.end(), 1.f);
}

Tensor Model::forward(const std::vector<int>& ids,
                      std::vector<Tensor>* activations) {
  const int n = (int)ids.size(), d = cfg.d_model;
  const int H = cfg.n_head, dh = d / H;

  // x[t] = token_emb[ids[t]] + pos_emb[t]
  Tensor x = zeros({n, d});
  for (int t = 0; t < n; t++)
    for (int i = 0; i < d; i++)
      x.at(t, i) = token_emb.at(ids[t], i) + pos_emb.at(t, i);

  if (activations) {
    activations->clear();
    activations->push_back(last_row(x));
  }

  for (int li = 0; li < cfg.n_layer; li++) {
    const LayerParams& p = layers[li];

    // -- attention block
    Tensor a = layernorm(x, p.ln1_w, p.ln1_b);
    Tensor q = matmul(a, p.wq), k = matmul(a, p.wk), v = matmul(a, p.wv);
    Tensor cat = zeros({n, d});
    Tensor attn_w;
    if (activations) attn_w = zeros({H, cfg.ctx, cfg.ctx});
    for (int h = 0; h < H; h++) {
      Tensor qh = slice_cols(q, h * dh, (h + 1) * dh);
      Tensor kh = slice_cols(k, h * dh, (h + 1) * dh);
      Tensor vh = slice_cols(v, h * dh, (h + 1) * dh);
      Tensor s = matmul(qh, transpose2d(kh)); // [n,n]
      const float inv = 1.f / std::sqrt((float)dh);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
          s.at(i, j) *= inv;
          if (j > i) s.at(i, j) = -1e9f; // causal mask
        }
      Tensor probs = softmax(s);
      if (activations)
        for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++)
            attn_w.data[(size_t)h * cfg.ctx * cfg.ctx + (size_t)i * cfg.ctx + j] =
                probs.at(i, j);
      Tensor ch = matmul(probs, vh); // [n,dh]
      for (int i = 0; i < n; i++)
        for (int j = 0; j < dh; j++) cat.at(i, h * dh + j) = ch.at(i, j);
    }
    x = add(x, matmul(cat, p.wo));

    if (activations) {
      activations->push_back(std::move(attn_w)); // part 0
      activations->push_back(last_row(x));       // part 1: post-attn residual
    }

    // -- mlp block
    Tensor m = layernorm(x, p.ln2_w, p.ln2_b);
    Tensor hidden = matmul(m, p.fc1);
    add_row_bias(hidden, p.b1);
    hidden = gelu(hidden);
    Tensor mlp = matmul(hidden, p.fc2);
    add_row_bias(mlp, p.b2);
    x = add(x, mlp);

    if (activations) {
      activations->push_back(last_row(hidden)); // part 2: mlp hidden
      activations->push_back(last_row(x));      // part 3: post-mlp residual
    }
  }

  Tensor xn = layernorm(x, lnf_w, lnf_b);
  Tensor logits = matmul(xn, transpose2d(token_emb)); // [n, vocab]
  return logits;
}

/* weights.bin: magic "KEYGPT01", i32 vocab/ctx/d_model/n_layer/n_head,
   then raw little-endian f32 tensors in README order. */

static bool write_tensor(std::ofstream& f, const Tensor& t) {
  f.write(reinterpret_cast<const char*>(t.data.data()),
          (std::streamsize)(t.data.size() * sizeof(float)));
  return f.good();
}

bool Model::save(const std::string& path) const {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write("KEYGPT01", 8);
  const int hdr[5] = {cfg.vocab, cfg.ctx, cfg.d_model, cfg.n_layer, cfg.n_head};
  f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
  write_tensor(f, token_emb);
  write_tensor(f, pos_emb);
  for (const auto& l : layers) {
    write_tensor(f, l.ln1_w);
    write_tensor(f, l.ln1_b);
    write_tensor(f, l.wq);
    write_tensor(f, l.wk);
    write_tensor(f, l.wv);
    write_tensor(f, l.wo);
    write_tensor(f, l.ln2_w);
    write_tensor(f, l.ln2_b);
    write_tensor(f, l.fc1);
    write_tensor(f, l.b1);
    write_tensor(f, l.fc2);
    write_tensor(f, l.b2);
  }
  write_tensor(f, lnf_w);
  write_tensor(f, lnf_b);
  return f.good();
}

static bool read_tensor(const uint8_t*& p, const uint8_t* end, Tensor& t) {
  const size_t bytes = t.data.size() * sizeof(float);
  if ((size_t)(end - p) < bytes) return false;
  std::memcpy(t.data.data(), p, bytes);
  p += bytes;
  return true;
}

bool Model::load_mem(const uint8_t* buf, size_t len) {
  const uint8_t* p = buf;
  const uint8_t* end = buf + len;
  if (len < 28 || std::memcmp(p, "KEYGPT01", 8) != 0) return false;
  p += 8;
  int hdr[5];
  std::memcpy(hdr, p, sizeof(hdr));
  p += sizeof(hdr);
  cfg.vocab = hdr[0];
  cfg.ctx = hdr[1];
  cfg.d_model = hdr[2];
  cfg.n_layer = hdr[3];
  cfg.n_head = hdr[4];
  init(cfg.vocab); // allocates shapes; values overwritten below
  if (!read_tensor(p, end, token_emb) || !read_tensor(p, end, pos_emb))
    return false;
  for (auto& l : layers) {
    if (!read_tensor(p, end, l.ln1_w) || !read_tensor(p, end, l.ln1_b) ||
        !read_tensor(p, end, l.wq) || !read_tensor(p, end, l.wk) ||
        !read_tensor(p, end, l.wv) || !read_tensor(p, end, l.wo) ||
        !read_tensor(p, end, l.ln2_w) || !read_tensor(p, end, l.ln2_b) ||
        !read_tensor(p, end, l.fc1) || !read_tensor(p, end, l.b1) ||
        !read_tensor(p, end, l.fc2) || !read_tensor(p, end, l.b2))
      return false;
  }
  return read_tensor(p, end, lnf_w) && read_tensor(p, end, lnf_b);
}

bool Model::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
  return load_mem(buf.data(), buf.size());
}
