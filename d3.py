import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# STEP 1: CHANGE THESE ONLY
# ============================================================

APP_TITLE = "RAG Q&A App"                           # <-- your app name
APP_CAPTION = "Data → Embeddings → RAG → Answers"   # <-- subtitle
SAMPLE_QUESTION = "What is the best product?"        # <-- placeholder in chat input
ROLE = "You are a data analyst assistant."           # <-- change role to match exam context

# ============================================================
# STEP 2: LOAD FAISS INDEX — no changes needed
# ============================================================

@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

db = load_db()

# ============================================================
# STEP 3: LLM + PROMPT — only change ROLE above if needed
# ============================================================

llm = ChatOpenAI(model='gpt-4o-mini', api_key=API_KEY)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"{ROLE}\n"
        "Answer ONLY using the provided context.\n"
        "Do not guess or hallucinate.\n"
        "If the answer is not present in the context reply exactly: "
        "'Not available in the provided data'"
    ),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# ============================================================
# STEP 4: STREAMLIT UI
# ============================================================

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"📊 {APP_TITLE}")
st.caption(APP_CAPTION)

# optional: show data preview if CSV exists
if os.path.exists("output.csv"):
    with st.expander("📋 Data Preview"):
        df = pd.read_csv("output.csv")
        st.dataframe(df.head(10), use_container_width=True)
elif os.path.exists("scraped_output.csv"):
    with st.expander("📋 Data Preview"):
        df = pd.read_csv("scraped_output.csv")
        st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# STEP 5: CHAT — no changes needed
# ============================================================

if "chat" not in st.session_state:
    st.session_state.chat = []  
for role, msg in st.session_state.chat:
    st.chat_message("user" if role == "user" else "assistant").write(msg)

# chat input
user_query = st.chat_input(SAMPLE_QUESTION)

if user_query:
    # show user message immediately
    st.chat_message("user").write(user_query)
    st.session_state.chat.append(("user", user_query))

    # similarity search
    top_k = 8
    threshold = 2.0
    docs_score = db.similarity_search_with_score(user_query, k=top_k)
    relevant_docs = [doc for doc, score in docs_score if score <= threshold]

    if not relevant_docs:
        answer = "Not available in the provided data"
    else:
        context = "\n\n".join(d.page_content for d in relevant_docs)
        response = llm.invoke(prompt.format(question=user_query, context=context))
        answer = response.content

        # citations
        answer += "\n\n**📎 Sources:**\n" + "\n".join(
            f"- {doc.page_content}" for doc in relevant_docs
        )

    st.chat_message("assistant").write(answer)
    st.session_state.chat.append(("bot", answer))