#include "model.h"
#include "tokenizer.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

// TODO(you): the training loop. skeleton is laid out in README order —
// works once you fill in the math. goal is to overfit corpus.txt.

static std::string read_file(const char* path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main() {
  const std::string corpus = read_file("corpus.txt");

  Tokenizer tok;
  tok.build(corpus);
  printf("vocab: %d chars, corpus: %zu bytes\n", tok.size(), corpus.size());

  std::vector<int> data = tok.encode(corpus);

  Model model;
  model.init(tok.size());

  // for iter in 0..N:
  //   sample a random window of ctx tokens from `data`
  //   logits = model.forward(window)
  //   loss = cross_entropy(logits[t], data[t+1]) — write this too
  //   backward + adam update
  //   every 100 iters: print loss
  //
  // then: model.save("weights.bin"); tok.save_vocab("vocab.txt");
  //
  // sanity check before wasm: greedy-generate from "Q: who are you\nA:"
  // natively and eyeball it. if it can't memorize here, it won't in wasm.

  (void)data;
  (void)model;
  printf("TODO: implement the training loop\n");
  return 0;
}
