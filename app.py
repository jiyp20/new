import os
import re
import json
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================================
# PATHS
# ==========================================

DATA_PATH = "data/cleaned_data.csv"
PDF_FOLDER_PATH = "data/pdfs"

df_global = pd.read_csv(DATA_PATH)

# ==========================================
# TOOL LOGGING
# ==========================================

tool_logs = []

def log_tool_call(name, inputs, output):
    tool_logs.append({
        "tool": name,
        "inputs": inputs,
        "output": output
    })

# ==========================================
# RAG SETUP
# ==========================================

def load_documents():
    docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    for filename in os.listdir(PDF_FOLDER_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER_PATH, filename)
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()

            split_docs = splitter.split_documents(pdf_docs)

            for doc in split_docs:
                doc.metadata["source"] = filename
                docs.append(doc)

    return docs

docs = load_documents()
embedding = OpenAIEmbeddings(api_key=API_KEY)
vector_db = FAISS.from_documents(docs, embedding)

# ==========================================
# ROUTING LOGIC (IMPROVED)
# ==========================================

def route_query(query):
    query_lower = query.lower()

    doc_keywords = [
        'policy', 'leave', 'notice', 'wfh',
        'work from home', 'sop', 'procedure',
        'onboarding', 'expense', 'grievance','budget','refund','benefits','perks',
        'overtime', 'training', 'remote','support','pricing','code of conduct'
    ]

    numeric_keywords = [
        'average', 'sum', 'total', 'count',
        'calculate', 'top', 'highest',
        'lowest', 'salary', 'performance',
        'hike', 'greater', 'less', 'above',
        'below', 'more than'
    ]

    doc_match = any(k in query_lower for k in doc_keywords)
    num_match = any(k in query_lower for k in numeric_keywords)

    if doc_match:
        return "RAG"
    elif num_match:
        return "TOOLS"
    else:
        return "REFUSE"

# ==========================================
# TOOLS
# ==========================================

@tool
def validate_query(query: str) -> dict:
    """
    Validate user query for unsupported, unsafe, or ambiguous requests.
    """

    if not query or len(query.strip()) == 0:
        return {"status": "blocked", "reason": "Empty query"}

    blocked_keywords = [
        "predict", "forecast", "future",
        "delete", "drop", "modify",
        "phone", "email", "address"
    ]

    ambiguous_words = [
        "maybe", "guess", "approximately",
        "around", "roughly"
    ]

    query_lower = query.lower()

    for word in blocked_keywords:
        if re.search(rf"\b{word}\b", query_lower):
            return {"status": "blocked", "reason": f"Unsupported: {word}"}

    for word in ambiguous_words:
        if word in query_lower:
            return {"status": "blocked", "reason": "Ambiguous query"}

    return {"status": "allowed"}


@tool
def calculate_stats(column: str, operation: str, department: str = None) -> dict:
    """
    Perform statistical operations: mean, sum, median, count, max, min.
    """

    df = df_global.copy()
    allowed_ops = ["mean", "sum", "median", "count", "max", "min"]

    if column not in df.columns:
        return {"status": "failed", "error": "Invalid column"}

    if operation not in allowed_ops:
        return {"status": "failed", "error": "Invalid operation"}

    if department:
        df = df[df["department"] == department]

    if df.empty:
        return {"status": "failed", "error": "No data"}

    value = getattr(df[column], operation)()

    result = {
        "status": "success",
        "result": round(float(value), 2)
    }

    log_tool_call("calculate_stats", locals(), result)
    return result


@tool
def group_analysis(group_by: str, agg_column: str, agg_func: str = "mean") -> dict:
    """
    Group by a column and aggregate.
    """

    df = df_global.copy()

    if group_by not in df.columns or agg_column not in df.columns:
        return {"status": "failed", "error": "Invalid columns"}

    grouped = df.groupby(group_by)[agg_column].agg(agg_func).to_dict()

    result = {
        "status": "success",
        "result": {k: round(float(v), 2) for k, v in grouped.items()}
    }

    log_tool_call("group_analysis", locals(), result)
    return result


@tool
def top_employees(n: int, sort_by: str) -> dict:
    """
    Retrieve top N employees sorted by numeric column.
    """

    df = df_global.copy()

    if sort_by not in df.columns:
        return {"status": "failed", "error": "Invalid column"}

    top_df = df.nlargest(n, sort_by)

    result = {
        "status": "success",
        "result": top_df[["employee_id", "department", sort_by]].to_dict("records")
    }

    log_tool_call("top_employees", locals(), result)
    return result

    """
    Count rows where numeric column satisfies a condition.
    Supported operators: >, <, >=, <=, ==
    """

    df = df_global.copy()

    if column not in df.columns:
        return {"status": "failed", "error": "Invalid column"}

    if operator == ">":
        filtered = df[df[column] > value]
    elif operator == "<":
        filtered = df[df[column] < value]
    elif operator == ">=":
        filtered = df[df[column] >= value]
    elif operator == "<=":
        filtered = df[df[column] <= value]
    elif operator == "==":
        filtered = df[df[column] == value]
    else:
        return {"status": "failed", "error": "Invalid operator"}

    result = {
        "status": "success",
        "count": len(filtered)
    }

    log_tool_call("conditional_count", locals(), result)
    return result


@tool
def apply_increment(employee_id: int = None, all_employees: bool = False) -> dict:
    """
    Apply salary revision based on slab rules.

    Rules:
    - base_salary < 40000 → 20% increment
    - 40000 ≤ base_salary ≤ 70000 → 10% increment
    - base_salary > 70000 → 5% increment
    """

    df = df_global.copy()

    def calculate(row):
        salary = row["base_salary"]

        if salary < 40000:
            increment_percent = 20
        elif 40000 <= salary <= 70000:
            increment_percent = 10
        else:
            increment_percent = 5

        revised_salary = salary * (1 + increment_percent / 100)

        return {
            "employee_id": int(row["employee_id"]),
            "old_salary": float(salary),
            "increment_percent": increment_percent,
            "revised_salary": round(float(revised_salary), 2)
        }

    if employee_id:
        emp = df[df["employee_id"] == employee_id]
        if emp.empty:
            return {"status": "failed", "error": "Employee not found"}

        result = {
            "status": "success",
            "result": calculate(emp.iloc[0])
        }

        log_tool_call("apply_increment", locals(), result)
        return result

    if all_employees:
        results = [calculate(row) for _, row in df.iterrows()]

        total_increment_cost = sum(
            r["revised_salary"] - r["old_salary"] for r in results
        )

        result = {
            "status": "success",
            "total_increment_cost": round(float(total_increment_cost), 2),
            "sample": results[:3]
        }

        log_tool_call("apply_increment", locals(), result)
        return result

    return {"status": "failed", "error": "Provide employee_id or all_employees=True"}


@tool
def apply_bonus(employee_id: int = None, all_employees: bool = False) -> dict:
    """
    Apply bonus eligibility rules.

    Bonus is applicable only if:
    - performance_rating >= 3

    Bonus amount is taken strictly from payroll dataset (monthly_bonus column).
    """

    df = df_global.copy()

    def calculate(row):
        performance = row["performance_rating"]
        monthly_bonus = row["monthly_bonus"]

        if performance >= 3:
            bonus_eligible = True
            eligible_bonus = monthly_bonus
        else:
            bonus_eligible = False
            eligible_bonus = 0

        return {
            "employee_id": int(row["employee_id"]),
            "performance_rating": float(performance),
            "bonus_eligible": bonus_eligible,
            "eligible_bonus": round(float(eligible_bonus), 2)
        }

    if employee_id:
        emp = df[df["employee_id"] == employee_id]
        if emp.empty:
            return {"status": "failed", "error": "Employee not found"}

        result = {
            "status": "success",
            "result": calculate(emp.iloc[0])
        }

        log_tool_call("apply_bonus", locals(), result)
        return result

    if all_employees:
        results = [calculate(row) for _, row in df.iterrows()]

        total_bonus_payout = sum(
            r["eligible_bonus"] for r in results
        )

        result = {
            "status": "success",
            "total_bonus_payout": round(float(total_bonus_payout), 2),
            "sample": results[:3]
        }

        log_tool_call("apply_bonus", locals(), result)
        return result

    return {"status": "failed", "error": "Provide employee_id or all_employees=True"}

# ==========================================
# AGENT SETUP
# ==========================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=API_KEY
)

tools = [
    validate_query,
    calculate_stats,
    group_analysis,
    top_employees,
    apply_increment,
    apply_bonus
]

system_prompt = """
You are an Enterprise AI Assistant.

You operate under STRICT enterprise rules.

GENERAL RULES:
1. ALWAYS call validate_query first.
2. If validation returns "blocked" → politely refuse.
3. NEVER compute numbers manually.
4. NEVER assume values.
5. Use ONLY payroll dataset for numeric operations.
6. If data is unavailable, respond:
   'Not available in the provided dataset.'
7. Do NOT hallucinate.
8. Do NOT answer from general knowledge.
9. Use tools whenever a query involves calculation, filtering, grouping, or business rules.

TOOL USAGE RULES:

• calculate_stats:
  Use for statistical operations such as mean, sum, median, count, max, min.

• group_analysis:
  Use for department-wise or month-wise grouped results.

• apply_increment:
  Use ONLY for salary revision based on salary slabs:
      - salary < 40000 → 20%
      - 40000 to 70000 → 10%
      - salary > 70000 → 5%

• apply_bonus:
  Use ONLY for bonus eligibility logic:
      - performance_rating >= 3 → eligible
      - performance_rating < 3 → not eligible
  Bonus must be taken strictly from payroll dataset (monthly_bonus column).

IMPORTANT:
- Never mix salary revision and bonus logic unless explicitly requested.
- Never guess counts.
- Never generate estimated values.
- If a request is outside scope, politely refuse.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# ==========================================
# RAG RESPONSE
# ==========================================

def rag_query(question):

    docs = vector_db.max_marginal_relevance_search(
        question,
        k=5,
        fetch_k=10
    )

    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))

    prompt_text = f"""
You are answering from internal company documents.

Use ONLY the relevant information from the context below.
Ignore unrelated sections.

If the answer is found, clearly summarize it.
If the answer is completely missing, respond exactly with:
Information not available in documents.

Context:
{context}

Question: {question}
"""

    response = llm.invoke(prompt_text)

    return f"{response.content}\n\nSources: {', '.join(sources)}"


# ==========================================
# MAIN QUERY HANDLER
# ==========================================

def ask_assistant(question: str):

    validation = validate_query.invoke({"query": question})

    if validation["status"] == "blocked":
        return f"Query blocked: {validation['reason']}"

    route = route_query(question)

    if route == "RAG":
        return rag_query(question)

    elif route == "TOOLS":
        result = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        return result["messages"][-1].content

    else:
        return "Unable to process this query. Please be more specific."


# ==========================================
# SAVE LOGS
# ==========================================

def save_logs():
    os.makedirs("logs", exist_ok=True)
    with open("logs/tool_logs.json", "w") as f:
        json.dump(tool_logs, f, indent=2)
    print("Logs saved to logs/tool_logs.json")
