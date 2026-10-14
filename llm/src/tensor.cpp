#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

// [m,k] x [k,n] -> [m,n]
Tensor matmul(const Tensor& a, const Tensor& b) {
  const int m = a.shape[0], k = a.shape[1], n = b.shape[1];
  Tensor out{{m, n}};
  out.data.assign((size_t)m * n, 0.f);
  for (int i = 0; i < m; i++)
    for (int kk = 0; kk < k; kk++) {
      const float av = a.at(i, kk);
      if (av == 0.f) continue;
      for (int j = 0; j < n; j++) out.at(i, j) += av * b.at(kk, j);
    }
  return out;
}

// rowwise softmax over the last dim — max subtracted for stability
Tensor softmax(const Tensor& t) {
  Tensor out = t;
  const int k = t.shape.back();
  const int rows = (int)t.data.size() / k;
  for (int r = 0; r < rows; r++) {
    float* row = out.data.data() + (size_t)r * k;
    const float mx = *std::max_element(row, row + k);
    float sum = 0.f;
    for (int j = 0; j < k; j++) sum += (row[j] = std::exp(row[j] - mx));
    for (int j = 0; j < k; j++) row[j] /= sum;
  }
  return out;
}

// normalize each row over the last dim, then affine w*x̂ + b
Tensor layernorm(const Tensor& x, const Tensor& w, const Tensor& b) {
  Tensor out = x;
  const int d = x.shape.back();
  const int rows = (int)x.data.size() / d;
  for (int r = 0; r < rows; r++) {
    float* row = out.data.data() + (size_t)r * d;
    float mean = 0.f;
    for (int j = 0; j < d; j++) mean += row[j];
    mean /= d;
    float var = 0.f;
    for (int j = 0; j < d; j++) {
      const float c = row[j] - mean;
      var += c * c;
    }
    const float inv = 1.f / std::sqrt(var / d + 1e-5f);
    for (int j = 0; j < d; j++)
      row[j] = (row[j] - mean) * inv * w.data[j] + b.data[j];
  }
  return out;
}

// tanh approximation — fine at this scale
Tensor gelu(const Tensor& x) {
  Tensor out = x;
  for (float& v : out.data) {
    const float u = 0.7978845608f * (v + 0.044715f * v * v * v);
    v = 0.5f * v * (1.f + std::tanh(u));
  }
  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = a;
  for (size_t i = 0; i < out.data.size(); i++) out.data[i] += b.data[i];
  return out;
}
