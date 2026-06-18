# LangChain ScyllaDB

This package contains the LangChain integration with [ScyllaDB Cloud](https://cloud.scylladb.com), backed by ScyllaDB's native HNSW vector index for high-performance Approximate Nearest Neighbour (ANN) search.

- `langchain-scylladb` ([PyPI](https://pypi.org/project/langchain-scylladb/))

## Features

### LangChain

#### Components

- [ScyllaDBVectorStore](#usage) — VectorStore backed by ScyllaDB Cloud's native `vector_index` (HNSW)

#### Highlights

- Native ANN search — no external vector database needed, vectors live alongside your data in ScyllaDB
- Application-side metadata filtering (`$and`, `$or`, `$not`, equality)
- Arbitrary string document IDs (not restricted to UUIDs)
- DC-aware load-balancing (required for ScyllaDB Cloud)
- Full [`VectorStoreIntegrationTests`](https://python.langchain.com/docs/contributing/how_to/integrations/standard_tests/) compliance (25/25 tests passing)

## Installation

```bash
pip install -U langchain-scylladb
```

## Usage

### ScyllaDBVectorStore

```python
import os
from langchain_scylladb import ScyllaDBVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = ScyllaDBVectorStore(
    embedding=OpenAIEmbeddings(),
    host=os.environ["SCYLLA_HOST"],
    username=os.environ["SCYLLA_USER"],
    password=os.environ["SCYLLA_PASSWORD"],
    local_dc=os.environ["SCYLLA_DC"],   # e.g. "AWS_US_EAST_1"
    keyspace="my_app",
    table_name="documents",
)

# Add documents
vectorstore.add_texts(
    ["ScyllaDB is a high-performance NoSQL database", "LangChain simplifies LLM apps"],
    metadatas=[{"source": "docs"}, {"source": "blog"}],
)

# Similarity search
docs = vectorstore.similarity_search("fast database", k=2)

# Filtered search
docs = vectorstore.similarity_search(
    "database", k=2, filter={"source": "docs"}
)
```

For more detailed usage examples, refer to the [ScyllaDB Cloud documentation](https://cloud.scylladb.com/docs).

## Contributing

See the [Contributing Guide](CONTRIBUTING.md).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
