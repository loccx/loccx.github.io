#pragma once

#include <string>
#include <vector>

// char-level tokenizer. vocab is built from the corpus; unknown chars map
// to the first id (used as an "unknown" slot — put a sentinel char first
// in vocab.txt).
struct Tokenizer {
  std::vector<int> stoi; // char -> id, indexed by byte value
  std::string itos;      // id -> char

  void build(const std::string& corpus);
  std::vector<int> encode(const std::string& s) const;
  std::string decode(const std::vector<int>& ids) const;

  // vocab.txt: one char per line, line number = id
  void save_vocab(const std::string& path) const;
  static Tokenizer load_vocab(const std::string& path);

  int size() const { return (int)itos.size(); }
};
