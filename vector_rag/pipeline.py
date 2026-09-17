import time
from typing import Any

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from requests.exceptions import ReadTimeout

from bootstrap import build_index
from config import API_KEY
from retriever import RAGRetriever

llm = ChatNVIDIA(
    model="nvidia/nemotron-nano-3-30b-a3b",
    nvidia_api_key=API_KEY,
    temperature=0.0,
    max_completion_tokens=1024,
)

classify_llm = ChatNVIDIA(
    model="nvidia/nemotron-nano-3-30b-a3b",
    nvidia_api_key=API_KEY,
    temperature=0.0,
    max_completion_tokens=1024,
)

GENERAL_SYSTEM_PROMPT = """You are an expert insurance advisor.
Answer the following general insurance question clearly and helpfully.
Use your general knowledge — you do NOT need to reference any specific policy document.

Question: {query}

Answer:"""


SPECIFIC_SYSTEM_PROMPT = """You are an expert insurance policy assistant. Answer the question using ONLY the provided context from the actual policy documents.

Rules:
- Use the context to provide a direct, factual answer
- If the context partially answers the question, provide what you can and note what is missing
- Only say "I don't have enough information" if the context is completely irrelevant to the question
- Be concise and precise
- Do not hallucinate or add information not present in the context"""


CLASSIFIER_PROMPT = """You are a query classifier for an insurance policy assistant.

Classify the query into exactly one of these categories:
- general: insurance-related conceptual, definitional, or comparative questions (what is health insurance?, explain copay, how does deductible work, difference between HMO and PPO). Must be about insurance or a related financial/health topic.
- specific: the user asks about a concrete provision, condition, claim, or coverage detail of a policy — the kind of thing you'd only find in the actual wording (grace periods, exclusions, claim process, deductibles, coverage limits).
- ambiguous: too vague to route (e.g. tell me about my policy).
- irrelevant: NOT related to insurance, policies, coverage, health plans, or financial protection at all (e.g. programming questions, cooking recipes, general trivia, math problems).

Respond with ONLY the category word (general, specific, ambiguous, or irrelevant), nothing else.

Query: {query}
Category:"""


def classify_query(query: str, llm: ChatNVIDIA = classify_llm, retries: int = 3) -> str:
    """Classify using classify_llm ONLY. Returns general, specific, or ambiguous."""
    for attempt in range(retries):
        try:
            response = llm.invoke(CLASSIFIER_PROMPT.format(query=query))
            category = response.content.strip().lower()
            if category not in ("general", "specific", "ambiguous"):
                # default to specific so we attempt retrieval
                return "specific"
            return category
        except ReadTimeout:
            if attempt < retries - 1:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2)
            else:
                print("Classifier timed out, defaulting to specific")
                return "specific"


def rag_query(
    query: str,
    retriever: RAGRetriever,
    llm: ChatNVIDIA = llm,
    top_k: int = 5,
    score_threshold: float = 0.2,
) -> dict[str, Any]:

    # guard empty query
    if not query or not query.strip():
        return {"answer": "Query cannot be empty", "sources": [], "context_used": False}

    # step 1 — classify using classify_llm ONLY
    category = classify_query(query)

    # step 1b — irrelevant: refuse to answer
    if category == "irrelevant":
        return {
            "answer": "I'm an insurance policy assistant and can only help with questions related to insurance, policies, coverage, or health plans. Please ask a relevant question.",
            "sources": [],
            "context_used": False,
            "category": category,
            "confidence": 1.0,
        }

    # step 2a — general: answer from LLM knowledge, no retrieval
    if category == "general":
        for attempt in range(3):
            try:
                response = llm.invoke(GENERAL_SYSTEM_PROMPT.format(query=query))
                return {
                    "answer": response.content,
                    "sources": [],
                    "context_used": False,
                    "category": category,
                    "confidence": 1.0,
                }
            except ReadTimeout:
                if attempt < 2:
                    print(f"General answer timeout, retrying attempt {attempt + 2}...")
                    time.sleep(3)
                else:
                    return {
                        "answer": "Generation timed out. Please try again.",
                        "sources": [],
                        "context_used": False,
                        "category": category,
                        "confidence": 0.0,
                    }

    # step 2b — ambiguous: ask for clarification
    if category == "ambiguous":
        return {
            "answer": "Your question is too vague. Could you specify which policy or coverage detail you're asking about?",
            "sources": [],
            "context_used": False,
            "category": category,
        }

    # step 2c — specific: retrieve from vector store
    results = retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)

    if not results:
        return {
            "answer": "I don't have enough information in the provided context to answer this.",
            "sources": [],
            "context_used": False,
            "category": category,
            "confidence": 0.0,
        }

    # step 3 — build context
    context_parts = []
    sources = []
    for i, doc in enumerate(results):
        context_parts.append(f"[{i + 1}] {doc['content']}")
        sources.append(
            {
                "id": doc["id"],
                "score": round(doc["similarity_score"], 4),
                "page": doc["metadata"].get("page", "unknown"),
                "source": doc["metadata"].get(
                    "source_file", doc["metadata"].get("source", "unknown")
                ),
                "preview": doc["content"][:150].strip() + "...",
            }
        )

    confidence = max([doc["similarity_score"] for doc in results])
    context = "\n\n".join(context_parts)

    prompt = f"""{SPECIFIC_SYSTEM_PROMPT}

    Context: {context}

    Question: {query}

    Answer:"""

    # step 4 — generate
    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            return {
                "answer": response.content,
                "sources": sources,
                "context_used": True,
                "retrieved_count": len(results),
                "category": category,
                "confidence": confidence,
            }
        except ReadTimeout:
            if attempt < 2:
                print(f"Generation timeout, retrying attempt {attempt + 2}...")
                time.sleep(3)
            else:
                return {
                    "answer": "Generation timed out. Please try again.",
                    "sources": sources,
                    "context_used": False,
                    "category": category,
                    "confidence": 0.0,
                }


if __name__ == "__main__":
    from retriever import RAGRetriever

    _, embedding_manager, vector_store = build_index()
    rag_retriever = RAGRetriever(
        vector_store=vector_store, embedding_manager=embedding_manager
    )

    for query in [
        "Why should i have a health insurance",
        "Write me a python program for Hello World",
        "Under what timeframe must the Insured submit a complete written claim to the Insurer after the Date of Loss for the claim to be payable?",
    ]:
        output = rag_query(query=query, retriever=rag_retriever, top_k=3)
        for key, value in output.items():
            print(f"{key}: {value}\n")
