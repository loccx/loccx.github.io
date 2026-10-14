#include "tensor.h"

#include <cstddef>

// TODO(you): implement each op. compile-checked stubs so `make` works today.

Tensor matmul(const Tensor& a, const Tensor& b) {
  // a: [m,k]  b: [k,n]  out: [m,n]
  Tensor out{{a.shape[0], b.shape[1]}};
  out.data.assign((size_t)out.shape[0] * out.shape[1], 0.f);
  return out;
}

Tensor softmax(const Tensor& t) {
  Tensor out = t;
  return out;
}

Tensor layernorm(const Tensor& x, const Tensor& w, const Tensor& b) {
  Tensor out = x;
  return out;
}

Tensor gelu(const Tensor& x) {
  Tensor out = x;
  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = a;
  return out;
}
