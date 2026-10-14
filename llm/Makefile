CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
EMCC ?= emcc

SRC := src/tokenizer.cpp src/tensor.cpp src/model.cpp

train: src/train.cpp $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

# outputs ../public/llm/llm.js + llm.wasm — the site loads it from there
wasm: src/wasm.cpp $(SRC)
	$(EMCC) -O2 -std=c++17 \
	  -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 \
	  -s ALLOW_MEMORY_GROWTH=1 \
	  -s EXPORTED_FUNCTIONS='["_llm_load","_llm_forward","_llm_logits","_llm_vocab_size","_llm_n_layers","_llm_activations","_llm_activations_len","_malloc","_free"]' \
	  -o ../public/llm/llm.js $^

clean:
	rm -f train ../public/llm/llm.js ../public/llm/llm.wasm weights.bin vocab.txt

.PHONY: wasm clean
