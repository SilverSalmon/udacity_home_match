# HomeMatch Project README

## Overview
HomeMatch is a Python/Jupyter Notebook application that generates synthetic real estate listings, stores them in a vector database, collects buyer preferences, and uses semantic search to match buyers with the most relevant listings. It also personalizes listing descriptions using an LLM.

## How to Run
1. Open `HomeMatch.ipynb` in Jupyter Notebook or VS Code.
2. Ensure you have an OpenAI API key and set it as the environment variable `OPENAI_API_KEY`.
3. Run each cell in order. The notebook will:
   - Generate 10 synthetic listings
   - Store them in ChromaDB
   - Collect and parse buyer preferences
   - Perform semantic search
   - Personalize listing descriptions
   - Save generated listings to a file named `listings`

## Dependencies
- Python 3.8+
- Jupyter Notebook
- openai
- langchain
- chromadb

Install dependencies with:
```
pip install langchain openai chromadb
```

## Files
- `HomeMatch.ipynb`: Main application notebook
- `listings`: File containing generated real estate listings
- `Project Instructions.md`: Project requirements

## Example Output
Example outputs are included in the notebook cells. You can modify buyer preferences and rerun the search to see different results.

## Stand-Out Features
- Easily extendable to support image search and multi-modal features using CLIP.
- Modular code for easy adaptation to other domains.

---

For any issues, please refer to the notebook comments or contact the project author.
