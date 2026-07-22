input_data: 
  path: "./data/input/OpenRagBench/pdf/arxiv/corpus/" 

hf:
  embeddings: "google/embeddinggemma-300m"
  llm_model: "lightning-ai/gpt-oss-20b"
query_settings:
  use: true  # Set to false to disable query rewriting
  type: "rewrite"  # Expand to "expansion", "hypothetical"

chunking:
  size: 500
  overlap: 100

vector_db:
  batch_size: 100
  persist_dir: "./chroma_db/OpenRagBench"