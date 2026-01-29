import os
import openai
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from unstructured.partition.pdf import partition_pdf
import importlib.metadata

# Load environment variables securely
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# -------- Agent State --------
class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    context: List[dict]
    intent_detected: bool
    step_count: int
    evaluation_score: float
    user_query: str
    retrieval_attempted: bool
    all_chunks: List[dict]


# -------- PDF Processing --------
def load_and_chunk_pdf(file_path: str) -> List[dict]:
    try:
        try:
            version = tuple(map(int, importlib.metadata.version("unstructured").split(".")))
            if version >= (0, 10, 0):
                elements = partition_pdf(filename=file_path, chunking_strategy="by_page", languages=["eng"])
            else:
                raise Exception
        except Exception:
            elements = partition_pdf(filename=file_path, languages=["eng"])
            pages = {}
            for el in elements:
                page = getattr(el.metadata, "page_number", 1)
                pages.setdefault(page, []).append(str(el))

            return [
                {"page": p, "text": "\n".join(pages[p])}
                for p in sorted(pages)
            ]

        return [
            {"page": getattr(el.metadata, "page_number", i), "text": str(el)}
            for i, el in enumerate(elements, 1)
        ]
    except Exception:
        return []


def load_documents(paths: List[str]) -> List[dict]:
    chunks = []
    for path in paths:
        chunks.extend(load_and_chunk_pdf(path))
    return chunks


# -------- Embeddings & Retrieval --------
def get_embedding(text: str) -> List[float]:
    try:
        res = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return res.data[0].embedding
    except Exception:
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def retrieve_chunks(query: str, chunks: List[dict], k: int = 3) -> List[dict]:
    query_emb = get_embedding(query)
    if not query_emb:
        return []

    scored = []
    for chunk in chunks:
        emb = get_embedding(chunk["text"])
        if emb:
            score = cosine_similarity(query_emb, emb)
            scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:k] if s > 0.2]


# -------- LLM Generation --------
def generate_answer(prompt: str) -> str:
    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "Unable to generate a response at the moment."


# -------- Agent Nodes --------
def intent_node(state: AgentState) -> dict:
    if not state["messages"]:
        return {"intent_detected": False}

    content = state["messages"][-1]["content"].strip()
    return {
        "intent_detected": bool(content),
        "user_query": content,
        "retrieval_attempted": False
    }


def retrieval_node(state: AgentState) -> dict:
    context = retrieve_chunks(state["user_query"], state["all_chunks"])
    return {
        "context": context,
        "retrieval_attempted": True
    }


def generation_node(state: AgentState) -> dict:
    context = state.get("context", [])
    query = state["user_query"]

    if not context:
        answer = "I could not find relevant information in the provided documents."
    else:
        context_text = "\n".join(c["text"] for c in context)
        prompt = f"""
        Use the context below to answer the question.
        If the answer is not present, say you don't know.

        Context:
        {context_text}

        Question:
        {query}
        """
        answer = generate_answer(prompt)

    return {"messages": [{"role": "assistant", "content": answer}]}


def evaluation_node(state: AgentState) -> dict:
    return {"evaluation_score": 1.0}


def decision_node(state: AgentState) -> str:
    if state["intent_detected"] and not state["retrieval_attempted"]:
        return "retrieve"
    return "end"


# -------- Workflow --------
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generation_node)
    graph.add_node("evaluate", evaluation_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        decision_node,
        {"retrieve": "retrieve", "end": END}
    )

    return graph.compile()


# -------- Main --------
if __name__ == "__main__":
    pdf_paths = []  # User provides paths at runtime
    all_chunks = load_documents(pdf_paths)

    agent = build_agent()
    state = {
        "messages": [],
        "context": [],
        "intent_detected": False,
        "step_count": 0,
        "evaluation_score": 0.0,
        "user_query": "",
        "retrieval_attempted": False,
        "all_chunks": all_chunks
    }

    while True:
        user_input = input("Query (type exit to quit): ").strip()
        if user_input.lower() == "exit":
            break

        state["messages"].append({"role": "user", "content": user_input})
        state.update(agent.invoke(state))

        print(state["messages"][-1]["content"])
