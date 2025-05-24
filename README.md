# HomeMatch Project README

## Overview
HomeMatch is a Python/Jupyter Notebook application that generates synthetic real estate listings, stores them in a vector database, collects buyer preferences, and uses semantic search to match buyers with the most relevant listings. It also personalizes listing descriptions using an LLM.

## How to Run
1. Open `HomeMatch.ipynb` in Jupyter Notebook or VS Code.
2. Ensure you have your .env file set up:
    OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
    OPENAI_API_EP = "https://reliasopenaitesting.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
    DALLE3_API_EP = "https://reliasopenaitesting.openai.azure.com/openai/deployments/Dalle3/images/generations?api-version=2024-02-01"
    DALLE3_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
    ADA_API_EP = "https://reliasopenaitesting.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
    ADA_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"


3. Run each cell in order. The notebook will:
   - Generate 10 synthetic listings
   - Store them in ChromaDB
   - Collect and parse buyer preferences
   - Perform semantic search
   - Personalize listing descriptions
   - Save generated listings to a file named `listings`

4. Run "streamlit run app.py" !!!!

## Dependencies
- Python 3.8+
- Jupyter Notebook
- openai
- langchain
- chromadb
- Streamlit

Install dependencies with:
```
pip install -r requirements.txt
```

## Files
- `HomeMatch.ipynb`: Main application notebook
- `listings`: File containing generated real estate listings
- `Project Instructions.md`: Project requirements
- `app.py`: Streamlit App!


## Example Output
Example outputs are included in the notebook cells. You can modify buyer preferences and rerun the search to see different results.

## Stand-Out Features
- Streamlit APP!
---

For any issues, please refer to the notebook comments or contact the project author.
