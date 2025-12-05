# r11: The Semantic Search Engine

![Rust](https://img.shields.io/badge/built_with-Rust-dca282.svg)
![Model](https://img.shields.io/badge/Model-MiniLM--L6--v2-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**r11** is a high-performance, local-first semantic search kernel built in Rust. 

Unlike traditional search engines that rely on exact keyword matching, r11 understands the **meaning** behind the query ("The Mind Reader"). It leverages state-of-the-art Vector Embeddings (BERT architecture) to map text into high-dimensional space, allowing for lightning-fast similarity comparisons.

## 🚀 Why r11?

* **Beyond Keywords:** Finds "canine" when you search for "dog." Matches intent, not just string literals.
* **Privacy-First:** All inference runs locally on the CPU using ONNX Runtime. No data leaves your machine.
* **Blazing Fast:** Built on Rust's zero-cost abstractions and the `fastembed` crate for optimal performance.
* **Lightweight:** Uses quantized models (~80MB) without sacrificing accuracy.

## 🛠 Architecture

r11 operates on the principle of **Vector Space**:

1.  **Ingestion:** Text inputs are tokenized and processed by the `all-MiniLM-L6-v2` transformer model.
2.  **Embedding:** The model outputs a normalized 384-dimensional dense vector.
3.  **Retrieval:** Similarity is calculated using Cosine Similarity (Dot Product for normalized vectors).

## 📦 Installation

Ensure you have Rust and Cargo installed.

```bash
git clone [https://github.com/jedizlapulga/r11.git](https://github.com/jedizlapulga/r11.git)
cd r11
cargo build --release
