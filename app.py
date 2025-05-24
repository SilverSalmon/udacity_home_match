import streamlit as st
import pandas as pd
import requests
import os
import chromadb
from uuid import uuid4
import ast
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_url = os.getenv("ADA_API_EP")
api_key = os.getenv("ADA_API_KEY")

# --- Embedding Function ---
def generate_embeddings(df, batch_size=1):
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    embeddings = []
    for i in range(0, len(df), batch_size):
        batch_texts = df.iloc[i:i + batch_size]["text"].fillna("").astype(str).tolist()
        payload = {"input": batch_texts}
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            batch_embeddings = [item["embedding"] for item in response.json()["data"]]
            embeddings.extend(batch_embeddings)
        else:
            embeddings.extend([[]] * len(batch_texts))
    result_df = df.copy()
    result_df["embeddings"] = embeddings
    return result_df

# --- Load Listings and Generate Embeddings ---
@st.cache_data(show_spinner=True)
def load_listings_and_embeddings():
    with open("listings.md", "r", encoding="utf-8") as file:
        listings_text = file.read()
    listings = [l.strip() for l in listings_text.strip().split("\n\n") if l.strip()]
    listings_df = pd.DataFrame({"listing": listings})
    listings_df = listings_df[listings_df['listing'].apply(len) >= 5].reset_index(drop=True)
    listings_embeddings_df = generate_embeddings(listings_df.rename(columns={"listing": "text"}), batch_size=1)
    return listings_embeddings_df

# --- Build ChromaDB Collection ---
def build_chroma_collection(listings_embeddings_df):
    client = chromadb.Client()
    # Always start fresh: delete if exists
    try:
        client.delete_collection("listings_collection")
    except Exception:
        pass  # Ignore if it doesn't exist
    collection = client.create_collection(name="listings_collection")
    for _, row in listings_embeddings_df.iterrows():
        collection.add(
            ids=[str(uuid4())],
            documents=[row["text"]],
            embeddings=[row["embeddings"]]
        )
    return collection

# --- Streamlit UI ---
st.title("🏡 HomeMatch: Real Estate Finder")
st.write("Enter your preferences and find the best matching homes!")

with st.spinner("Loading listings and generating embeddings..."):
    listings_embeddings_df = load_listings_and_embeddings()
    collection = build_chroma_collection(listings_embeddings_df)

st.header("Enter Your Preferences")
def_pref = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]

questions = [
    "How big do you want your house to be?",
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]

user_answers = []
for i, q in enumerate(questions):
    ans = st.text_input(q, value=def_pref[i])
    user_answers.append(ans)

if st.button("Find My Home!"):
    buyer_profile = "\n".join(user_answers)
    # Generate embedding for buyer profile
    headers = {"Content-Type": "application/json", "api-key": api_key}
    data = {"input": buyer_profile}
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        buyer_embedding = response.json()["data"][0]["embedding"]
        results = collection.query(query_embeddings=[buyer_embedding], n_results=3)
        st.subheader("Top Matching Listings:")
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0]), 1):
            st.markdown(f"**Result {i}:**")
            st.text(doc)
            st.markdown(f"Distance Score: `{dist:.4f}`")
            # --- Personalized Description via LLM ---
            # Use Azure OpenAI for personalization
            import openai
            from openai import AzureOpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://reliasopenaitesting.openai.azure.com/")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            try:
                client = AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=openai_key,
                )
                prompt = f"""
Given the following real estate listing and buyer preferences, rewrite the listing description to emphasize features that match the preferences. Do not change factual details.\n\nBuyer Preferences:\n{buyer_profile}\n\nListing:\n{doc}\n\nPersonalized Description:
"""
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful real estate assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    temperature=0.7,
                    top_p=1.0,
                    model=deployment
                )
                personalized_desc = resp.choices[0].message.content.strip()
                st.markdown(f"**Personalized Description:**\n{personalized_desc}")
            except Exception as e:
                st.error(f"Error generating personalized description: {e}")
                personalized_desc = doc
            # --- DALL-E 3 Image Generation ---
            # Use the same approach as in home_match.ipynb
            dalle3_ep = os.getenv("DALLE3_API_EP")
            dalle3_key = os.getenv("DALLE3_API_KEY")
            dalle_headers = {
                "Content-Type": "application/json",
                "api-key": dalle3_key
            }
            dalle_payload = {
                "prompt": f"High-resolution photo of realestate for sale, Photographed in natural daylight like a real estate listing photo. {personalized_desc}",
                "n": 1,
                "size": "1024x1024"
            }
            image_url = None
            if dalle3_ep and dalle3_key:
                try:
                    dalle_resp = requests.post(dalle3_ep, headers=dalle_headers, json=dalle_payload)
                    if dalle_resp.status_code == 200:
                        data = dalle_resp.json()
                        image_url = data['data'][0]['url']
                        st.image(image_url, caption="Generated Home Image", width=384)
                    else:
                        st.warning(f"DALL-E 3 image generation failed: {dalle_resp.status_code} - {dalle_resp.text}")
                except Exception as e:
                    st.warning(f"DALL-E 3 image generation error: {e}")
            else:
                st.info("DALL-E 3 API endpoint/key not set. Skipping image generation.")
            st.markdown("---")
    else:
        st.error(f"Error generating embedding: {response.status_code} - {response.text}")
