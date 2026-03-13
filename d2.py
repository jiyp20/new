import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("scraped.csv")      
print(df.columns.tolist())             
print(df.head(3))

def row_to_text(row):

    # --- CASE A: product/dummyjson style ---
    # return (
    #     f"Product: {row['title']} | "
    #     f"Category: {row['category']} | "
    #     f"Brand: {row['brand']} | "
    #     f"Price: {row['price']} | "
    #     f"Rating: {row['rating']} | "
    #     f"Stock: {row['stock']}"
    # )

    # --- CASE B: covid style ---
    # return (
    #     f"Date: {row['date']} | "
    #     f"Confirmed: {row['confirmed_total']} | "
    #     f"Recovered: {row['recovered_total']} | "
    #     f"Deceased: {row['deceased']} | "
    #     f"Tested: {row['tested_total']}"
    # )

    # --- CASE C: scraping/laptop style ---
    return (
        f"Product: {row['product_name']} | "
        f"Price: {row['price']} | "
        f"Description: {row['description']} | "
        f"Reviews: {row['review']}"
    )

    # --- CASE D: unknown — just use ALL columns automatically ---
    # return " | ".join(f"{col}: {row[col]}" for col in df.columns)


documents = [
    Document(
        page_content=row_to_text(row),
        metadata={"index": str(i)}
    )
    for i, (_, row) in enumerate(df.iterrows())
]

print(f"Documents created: {len(documents)}")
print("Sample:", documents[0].page_content)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=30
)

chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")


embeddings = OpenAIEmbeddings(api_key=API_KEY)
db = FAISS.from_documents(chunks, embeddings)
print("FAISS index ready")

db.save_local("faiss_index")

query = "high rating affordable product"  

results = db.similarity_search(query, k=5)

print(f"\nTop {len(results)} results for query: '{query}'")
print("-" * 50)
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)