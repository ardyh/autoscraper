MrScraper — HTML to JSON Parser (LLM Engineer Take-Home)
=======================================================

Overview
--------
An API that parses raw HTML and returns structured JSON based on a natural language query. The API runs locally using a self-hosted model via Ollama (no paid external APIs).

Key features:
- Python FastAPI service exposing `POST /parse`
- Accepts `{ html: string, query: string }`
- Returns structured JSON plus metadata
- Default Ollama model: `llama3` (configurable via `MODEL_ID`)

Implementation
--------------------
- Model: uses local Ollama HTTP API. Default `MODEL_ID=llama3`.
- Approach: **Sliding Window RAG (Retrieval Augmented Generation)**
  - HTML is converted to rich text (preserving links, images, data attributes)
  - Content is chunked and indexed using semantic embeddings
  - Multiple non-overlapping windows are retrieved and processed
  - Results are deduplicated for final output
- Fallback: set `USE_SIMPLE_PARSER=1` to use direct LLM approach (no RAG)
- Postprocessing: the service attempts to parse JSON from the model output.
- No paid external APIs are used (all processing is local).

Notes on Performance and Limitations
------------------------------------
Strengths:
- **Generalizability:** This approach does not depend on hard-coded parsing for a single use-case. We can also easily swap the model used for generation and the DB used for retrieval
- **Coverage:** The retriever basically covers the whole HTML page for search
- **Scalability:** I believe that using this approach, a cost effective way to increase performance and reduce latency is to just use a bigger model with longer context size and more parameters. Latency will decrease as we reduce the number of iteration, and accuracy will increase as we're supposed to use a 'smarter' model
Weaknesses:
- **Latency:** On average, the provided scenarios took >2m to run. The blocker is that we need to perform the "generation" part multiple times, and the length of context that we feed the LLM is also not short. Although we need to take into account that this is run on a consumer device. Assuming it's done on a proper server, we might achieve 3x speedup
- **Accuracy:** 
  - Since we're dividing the content into blocks, there's a chance that the information for one item might not be included within the same block, therefore leading the LLM to output `null` in some fields. For example, the information on `"job": "nurse"` lives on separate blocks from the information on its salary. 
  - Since we're leaving the JSON generation part to the LLM, there's a chance that the LLM doesn't understand the context provided and therefore outputs `null` to some fields, or that some items have different field compared to others. For example: one item has the field `companyName` while another doesn't
- **Tuning:** On some scenarios, I needed to tune the number of retrieved context window to feed to the LLM

Model and Architecture Choice
----------
Overall, I tried ~7 approaches, documented within the notebooks/ directory.
- First, I tried feeding the HTML directly to the LLM. This quickly failed due to limited context size. 
- Then, I tried simple RAG using TFIDF and BM50. I stayed on this approach for quite a bit, but then I moved on as I saw that the LLM couldn't take in raw HTML as context and output a coherent JSON. Those simple search systems also lacked the capability to search for highly specific information
- Then, I tried combining the simple RAG above with code generation, meaning that the LLM can output a scraping script (BeautifulSoup) and scrape the HTML automatically. But then I realized that the scripts generated are too simple, and therefore cannot properly extract content from the HTML. 
- Then, I tried creating an "agent", which can decide for itself how to explore the document and create a script to extract content from it. This is a minified version of a tool calling agent like we have on Claude and ChatGPT. But I discovered that the code and exploration that it does is still too simple, likely due to lacking parameters. I also found that over iterations, it doesn't really improve on its extraction code, so I saw this as underperforming
- Then, I moved to Langchain RAG, using the Chroma DB for vector search and a HuggingFace model for the embedding. This is when I started noticing some good results
- I tried integrating RAG with code generation, such that the system can search using a sophisticated search method (vector search) and then produce code to scrape the information. But as I found before, the generated scraping script was still too simple. 
- So I went back to pure RAG, and I found out that the LLM hasn't been given much information from the HTML as we only stored text on our vector database. So we modified the HTML parser to include metadata like classes and HTML attributes to store within the vector DB. This then increased performance significantly. I also implemented a rolling window such that the LLM can "scroll over" the HTML page blocks that's been stored on the vector DB

Environment (Conda)
-------------------
1) Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate mrscraper
```

2) Install and run Ollama (one-time):

```bash
# macOS (see https://ollama.com)
brew install --cask ollama
ollama serve &
# Pull a model (choose one):
ollama pull llama3   # default
# ollama pull mistral
```

3) Install jq (for curl examples):

```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt-get install jq

# Or skip this if you'll use Python test scripts instead
```

4) Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5) Optional: Configure environment variables

```bash
export MODEL_ID=llama3                          # Ollama model (default: llama3)
export OLLAMA_BASE_URL=http://localhost:11434   # Ollama API URL (default)
export OLLAMA_TIMEOUT=600                       # Request timeout in seconds (default: 600)
export USE_SIMPLE_PARSER=0                      # Set to 1 to disable RAG (default: 0)
```

**I stored the output of all scenarios in `data/outputs`**

API
---
- `POST /parse`
  - Request body:

    ```json
    {
      "html": "<html>…</html>",
      "query": "Extract job title, location, salary, and company name"
    }
    ```

  - Response body (example):

    ```json
    {
      "data": [
        {"title": "…", "location": "…", "salary": "…", "company": "…"}
      ],
      "meta": {
        "model": "llama3",
        "elapsed_ms": 5234,
        "approach": "sliding_window_rag",
        "chunks_examined": 90,
        "items_extracted": 10
      }
    }
    ```

Usage Examples
--------------
All scenarios send raw HTML to the API. Below, we fetch HTML then call the API.

**Note:** These examples use `jq` to properly escape HTML for JSON. Install with: `brew install jq` (macOS) or `apt-get install jq` (Linux).

Scenario 1 — Books to Scrape (name, price):

```bash
curl -s https://books.toscrape.com/ | \
jq -Rs --arg q "Return the books: name and price" '{html: ., query: $q}' | \
curl -s -X POST http://localhost:8000/parse \
  -H 'Content-Type: application/json' \
  -d @- | jq '.'
```

Or using the helper script:

```bash
./test_api_curl.sh "https://books.toscrape.com/" "Return the books: name and price"
```

Scenario 2 — Job Listings (title, location, salary, company):

```bash
curl -s 'https://medrecruit.medworld.com/jobs/list?location=New+South+Wales&page=1' | \
jq -Rs --arg q "Extract job title, location, salary, and company name from the listings" '{html: ., query: $q}' | \
curl -s -X POST http://localhost:8000/parse \
  -H 'Content-Type: application/json' \
  -d @- | jq '.'
```

Scenario 3 — Club Listing (club name, logo, website):

```bash
curl -s https://www.azsoccerassociation.org/member-clubs/ | \
jq -Rs --arg q "Get the club names, logo image links and their official websites" '{html: ., query: $q}' | \
curl -s -X POST http://localhost:8000/parse \
  -H 'Content-Type: application/json' \
  -d @- | jq '.'
```

Scenario 4 — Hidden Info (name, address, lat/lng):

```bash
curl -s 'https://avantstay.com/503003/park-city/silverado?adults=1' | \
jq -Rs --arg q "Return the property name, address, latitude and longitude" '{html: ., query: $q}' | \
curl -s -X POST http://localhost:8000/parse \
  -H 'Content-Type: application/json' \
  -d @- | jq '.'
```

**Alternative:** Use the Python test script for easier testing:

```bash
python3 test_api_simple.py "https://books.toscrape.com/" "Return the books: name and price"
```

Development
-----------
Project layout:

```
app/
  main.py                # FastAPI app
  schemas.py             # Request/response models
  llm/model.py           # Ollama client wrapper
  service/
    parser.py            # Main entry point (routes to RAG or simple)
    rag_parser.py        # Sliding window RAG implementation
  utils/html.py          # Smart HTML extraction + utilities

test_api.py              # Test suite for local HTML files
test_api_simple.py       # Test script for live URLs (Python)
test_api_curl.sh         # Test script for live URLs (bash + jq)

notebooks/
  iterative_rag_extraction.ipynb  # RAG exploration and development
  
API_USAGE.md             # Detailed API documentation
CHANGES.md               # Changelog and migration guide
```

Run tests:

```bash
# Test all scenarios with local HTML files
python test_api.py

# Test with live URLs (Python)
python3 test_api_simple.py "https://books.toscrape.com/" "Return the books: name and price"

# Test with live URLs (bash + jq)
./test_api_curl.sh "https://books.toscrape.com/" "Return the books: name and price"
```

License
-------
MIT


