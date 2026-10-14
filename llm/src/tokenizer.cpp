#include "tokenizer.h"

#include <algorithm>
#include <fstream>
#include <set>

// complete — tokenization is plumbing, not the lesson.

void Tokenizer::build(const std::string& corpus) {
  std::set<char> chars(corpus.begin(), corpus.end());
  itos.assign(chars.begin(), chars.end()); // sorted: deterministic
  stoi.assign(256, -1);
  for (int i = 0; i < (int)itos.size(); i++) stoi[(unsigned char)itos[i]] = i;
}

std::vector<int> Tokenizer::encode(const std::string& s) const {
  std::vector<int> out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    int id = stoi[c];
    out.push_back(id >= 0 ? id : 0); // 0 = unknown
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  for (int id : ids) out += itos[id];
  return out;
}

void Tokenizer::save_vocab(const std::string& path) const {
  std::ofstream f(path);
  for (char c : itos) f << (c == '\n' ? "\\n" : std::string(1, c)) << '\n';
}

Tokenizer Tokenizer::load_vocab(const std::string& path) {
  Tokenizer t;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    t.itos += line == "\\n" ? '\n' : line[0];
  }
  t.stoi.assign(256, -1);
  for (int i = 0; i < (int)t.itos.size(); i++)
    t.stoi[(unsigned char)t.itos[i]] = i;
  return t;
}
