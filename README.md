# Agentic RAG — Practical experimentation with retrieval-augmented generation

Please read full article on medium: [Build Agentic RAG using LangGraph](https://medium.com/@alphaiterations/build-agentic-rag-using-langgraph-b568aa26d710)








This project is an interactive notebook-driven exploration of Retrieval-Augmented Generation (RAG) patterns for medical knowledge and device manuals. It demonstrates a traditional RAG pipeline and an "agentic" RAG variant where a lightweight routing agent chooses among multiple retrieval tools (ChromaDB collections and a web search) before generating an answer with an LLM.

The content below is distilled from the notebook `agnetic-rag.ipynb` and prepared for publication (for example, on Medium). It preserves the original structure: dataset preparation, vector store setup, simple LLM wiring, a traditional RAG workflow, and an agentic RAG workflow with routing and relevance-checking.

---

## Motivation

RAG systems combine external knowledge with generative models to produce grounded, up-to-date, and more accurate responses. In domains like healthcare and medical devices, grounding answers in trusted documentation (Q&A datasets and device manuals) and selective web search can improve reliability and context-awareness.

This notebook explores two approaches:
- Traditional RAG: single retriever -> prompt builder -> LLM.
- Agentic RAG: routing agent chooses the best retriever (Q&A collection, device manuals collection, or web search); retrieved context is validated for relevance and iterated if needed.

---

## Datasets used

Two CSV datasets are used in the experiments (both sampled down to 500 rows for quick iteration):

- `medical_q_n_a.csv` — a comprehensive medical question-and-answer dataset. The notebook samples 500 rows and combines each row into a single `combined_text` field like: `Question: ... Answer: ... Type: ...`.
- `medical_device_manuals_dataset.csv` — medical device manuals and metadata. The notebook samples 500 rows and creates `combined_text` like: `Device Name: ... Model: ... Manufacturer: ... Indications: ... Contraindications: ...`.

These combined text fields are used as documents for the vector store.

---

## Components and libraries

The experiments use a small set of Python libraries and tools (shown here as used in the notebook):

- pandas, numpy — data handling
- sentence-transformers / chroma default embeddings — document embeddings
- chromadb (PersistentClient) — vector store
- langchain_community.utilities.GoogleSerperAPIWrapper — lightweight web search
- openai (OpenAI client) — LLM calls
- langgraph (StateGraph) — simple graph-based workflow agent
- dotenv — environment variable loading

The notebook also references FAISS, scikit-learn metrics, seaborn, and the SentenceTransformer model for embedding experiments.

Note: The exact package versions are not included in the notebook. If you reproduce this, pin versions for reproducibility.

---

## Setup (how the notebook is structured)

1. Load environment variables with `dotenv` (expecting `OPEN_AI_KEY` and `SERPER_API_KEY`).
2. Read and sample the two CSV datasets. Create a `combined_text` column for each dataset to form each document.
3. Initialize a ChromaDB persistent client and create two collections: `medical_q_n_a` and `medical_device_manual`.
4. Add sampled documents, their metadata, and string ids to the respective collections.
5. Test simple retrieval queries against each collection.
6. Initialize a web search wrapper (`GoogleSerperAPIWrapper`) for external queries.
7. Create a small LLM wrapper, `get_llm_response(prompt)`, that calls the OpenAI client and returns the assistant/chat response.

The notebook keeps things intentionally small (500-row samples) so experiments run quickly.

---

## Traditional RAG workflow (Part A)

The traditional RAG agent in the notebook is implemented as a tiny workflow using `langgraph.StateGraph` with three steps:

1. Retriever: run a similarity query against the `medical_q_n_a` collection and join the top documents into a single context string.
2. PromptBuilder: create a simple RAG prompt that injects the retrieved context and the user question, with a length constraint.
3. LLM: call `get_llm_response(prompt)` and save the model output in the workflow state.

This pipeline is compiled into a workflow graph and run against example queries like `What are the treatments for Kawasaki disease ?`.

---

## Agentic RAG workflow (Part B)

The core idea: instead of always using the same retriever, use a small routing agent that decides among multiple retrieval options and then validates the retrieved context for relevance. Key pieces:

- Router node: constructs a short decision prompt and asks the LLM to return one of three labels: `Retrieve_QnA`, `Retrieve_Device`, or `Web_Search`.
- Retriever nodes: one for the Q&A Chroma collection, one for the device manuals Chroma collection, and one for the web search API. Each returns a context string and sets a `source` label.
- Relevance checker: asks the LLM whether the retrieved context is relevant (answer should be `Yes` or `No`).
- Conditional routing: if relevance is `Yes`, continue to prompt building and LLM. If `No`, fall back to `Web_Search` and re-run the relevance check; the flow limits iterations to a maximum of 3 attempts.

The agentic flow composes these nodes into a StateGraph with conditional edges and compiles it into an executable agent called `agentic_rag` in the notebook. The notebook visualizes the flow with `draw_mermaid_png()` and runs example queries against the agent.

---

## Example code snippets (conceptual)

These are concise conceptual snippets that mirror the notebook's structure and can be adapted for production or a blog post. They are intentionally simplified for readability.

1) LLM wrapper

```
def get_llm_response(prompt: str) -> str:
    client_llm = OpenAI(api_key=openai_api_key)
    response = client_llm.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

2) Simple retriever node (Chroma query)

```
results = collection1.query(query_texts=[query], n_results=3)
context = "\n".join(results["documents"][0])
```

3) Router decision prompt (asks LLM to choose the best retriever)

```
You are a routing agent. Based on the user query, decide where to look for information.

Options:
- Retrieve_QnA: if it's about general medical knowledge, symptoms, or treatment.
- Retrieve_Device: if it's about medical devices, manuals, or instructions.
- Web_Search: if it's about recent news, brand names, or external data.

Query: "{query}"

Respond ONLY with one of: Retrieve_QnA, Retrieve_Device, Web_Search
```

---

## Observations & tips

- Agentic routing allows you to compose multiple specialized knowledge sources and route queries dynamically, which can improve grounding for domain-specific queries.
- Use deterministic routing (few-shot prompts, stricter instruction) or a small classification head if you need more consistent routing decisions than a freeform LLM response.
- Limit iterations and fallbacks to avoid long or costly loops (this notebook uses a 3-iteration limit).
- Always validate LLM-provided labels and guard against hallucinated routing decisions by checking the responses format (strict text constraint or model-based classifier).

---

## How to reproduce

1. Clone the repository and open `agnetic-rag.ipynb` in Jupyter or VS Code.
2. Create a `.env` file with:

```
OPEN_AI_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

3. Install required packages (pin versions in your environment). At minimum you'll need `pandas`, `chromadb`, `openai`, `langchain_community`, `sentence-transformers`, `langgraph`, and `python-dotenv`.
4. Run the notebook cells in order. The notebook uses small samples (500 rows) so it should run quickly on a laptop.

---

## Next steps / Improvements

- Add explicit dependency management (requirements.txt or pyproject.toml) and pin package versions.
- Add unit tests for the routing logic and the relevance-checker decision function.
- Replace freeform LLM routing with a small supervised classifier for stability in production.
- Add logging, metrics, and provenance (which document produced which facts) for auditability in medical domains.
- Add visualization assets (Mermaid diagrams exported as images) for the Medium article.

---

## Closing

This notebook is a practical, experiment-focused exploration of RAG and agentic RAG for medical content. It demonstrates how a tiny routing agent and conditional workflows can make RAG systems more flexible and context-aware while keeping iteration costs bounded.

If you'd like, I can also:
- Extract runnable scripts from notebook cells (python modules) and add a requirements file.
- Prepare a Medium-ready article with images and code highlights derived from this README.
# agentic-rag
Agentic RAG
