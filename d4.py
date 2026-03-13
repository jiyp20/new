import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("output.csv")  # change filename if needed

# ---- TOOL 1 ----
@tool
def get_stats(column: str) -> str:
    """Get average, max, min, total of a numeric column. Input: column name."""
    if column not in df.columns:
        return f"Column not found. Available: {df.columns.tolist()}"
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"'{column}' is not numeric."
    return (f"Average: {df[column].mean():.2f} | Max: {df[column].max():.2f} | "
            f"Min: {df[column].min():.2f} | Total: {df[column].sum():.2f}")

# ---- TOOL 2 ----
@tool
def filter_rows(condition: str) -> str:
    """Filter rows using pandas query. Input: condition e.g. 'price < 500' or 'rating > 4'"""
    try:
        result = df.query(condition)
        if result.empty:
            return "No rows matched."
        return result.head(10).to_string(index=False)
    except Exception as e:
        return f"Error: {e}"

# ---- TOOL 3 ----
@tool
def validate_query(question: str) -> str:
    """
    Always call this first. Checks if question is relevant to the dataset.
    Blocks irrelevant queries like weather, predictions, news.
    """
    blocked = ["weather", "predict", "forecast", "news", "sports", "movie", "delete"]
    for kw in blocked:
        if kw in question.lower():
            return f"BLOCKED: '{kw}' is outside dataset scope. Refuse politely."
    return f"ALLOWED. Columns available: {df.columns.tolist()}"

# ---- AGENT ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=API_KEY)

tools = [validate_query, get_stats, filter_rows]

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data analyst assistant.\n"
     "ALWAYS call validate_query first.\n"
     "If BLOCKED → refuse politely.\n"
     "NEVER compute numbers yourself — use tools.\n"
     "If not in data say: 'Not available in the provided data.'"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# ---- TEST ----
queries = [
    "What is the average price?",
    "Show products with rating above 4.5",
    "What is the weather today?",       # BLOCKED
]

for q in queries:
    print(f"\nQ: {q}")
    result = agent_executor.invoke({"input": q})
    print(f"A: {result['output']}")
    print("-" * 50)