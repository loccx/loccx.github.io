#pragma once

#include <vector>

// a tensor is just floats + a shape. keep it this simple — you can feel
// every allocation.
struct Tensor {
  std::vector<int> shape;   // e.g. {d}, {n,d}, {h,t,t}
  std::vector<float> data;

  float& at(int i) { return data[i]; }
  float& at(int i, int j) { return data[i * shape[1] + j]; }
  float at(int i, int j) const { return data[i * shape[1] + j]; }
};

// TODO(you): the four primitives the model needs. signatures are set;
// implement them in tensor.cpp. these are the whole lesson's foundation —
// write them slowly and test each one on tiny inputs before moving on.

Tensor matmul(const Tensor& a, const Tensor& b); // [m,k] x [k,n] -> [m,n]
Tensor softmax(const Tensor& t);                 // last dim, subtract max first
Tensor layernorm(const Tensor& x, const Tensor& w, const Tensor& b);
Tensor gelu(const Tensor& x);
Tensor add(const Tensor& a, const Tensor& b);    // elementwise, same shape
