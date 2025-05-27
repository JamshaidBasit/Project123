from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from typing import TypedDict

# ─── API KEY ─────────────────────────────────────────────────────────────
groq_api_key = "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"
# --- CHANGE THIS LINE ---
llm = ChatGroq(temperature=0, model_name="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=groq_api_key)
# ------------------------

# ─── STATE SCHEMA ────────────────────────────────────────────────────────
class State(TypedDict):
    input: str
    output: str
    confidence: int
    retries: int
    human_review: bool

# ─── AGENT STEP ──────────────────────────────────────────────────────────
def agent_step(state: State) -> State:
    input_text = state["input"]
    prompt = f"Answer this clearly:\n{input_text}"
    print(f"\n🚀 AGENT STEP - Attempt {state.get('retries', 0) + 1}")
    print("Prompt:", prompt)
    response = llm.invoke(prompt)
    print("Agent's Output:", response.content) # Print the output here
    return {
        **state,
        "output": response.content
    }

# ─── EVALUATION STEP ─────────────────────────────────────────────────────
def evaluation_step(state: State) -> State:
    prompt = f"Answer this clearly and factually:\n{state['input']}"
    output_text = state["output"]

    # ─── Print Input & Output ──────────────────────────────
    print(f"\n📝 EVALUATION STEP - Attempt {state.get('retries', 0) + 1}")
    print("Prompt:", prompt)

    eval_prompt = f"""
Evaluate the following:

Prompt:
{prompt}

Response:
"{output_text}"

How correct and relevant is the Response to the Prompt?

Give a confidence score between 0 and 100.

Respond in JSON format:
{{"confidence": <score>}}
"""
    eval_response = llm.invoke(eval_prompt).content

    try:
        confidence = int(eval_response.split('"confidence":')[1].split("}")[0].strip())
    except Exception:
        confidence = 50  # Default fallback

    return {
        **state,
        "confidence": confidence,
        "retries": state.get("retries", 0) + 1 # Increment retries
    }

# ─── CONDITIONAL EDGE FUNCTION ───────────────────────────────────────────
def route_next_step(state: State) -> str:
    if state["confidence"] >= 98:
        print(f"\n✨ Confidence {state['confidence']}% is sufficient. Ending process.")
        return "end"
    elif state["retries"] < 3:
        print(f"\n🔄 Confidence {state['confidence']}% is too low. Retrying... (Attempt {state['retries'] + 1} of 3)")
        return "agent" # Re-run the agent if confidence is low and retries left
    else:
        print(f"\n⚠️ Confidence {state['confidence']}% still too low after 3 retries. Human review needed.")
        return "human_review_needed" # Go to a final state indicating human review

# ─── HUMAN REVIEW STEP ───────────────────────────────────────────────────
def human_review_step(state: State) -> State:
    return {
        **state,
        "human_review": True
    }

# ─── GRAPH BUILDING ──────────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("agent", RunnableLambda(agent_step))
builder.add_node("evaluation", RunnableLambda(evaluation_step))
builder.add_node("human_review_needed", RunnableLambda(human_review_step)) # New node for human review

builder.set_entry_point("agent")
builder.add_edge("agent", "evaluation")

# Add conditional edge from evaluation
builder.add_conditional_edges(
    "evaluation",
    route_next_step,
    {
        "agent": "agent",  # Loop back to agent
        "human_review_needed": "human_review_needed", # Go to human review
        "end": END
    }
)

# Edge to the final human review state
builder.add_edge("human_review_needed", END)

graph = builder.compile()

# ─── SAMPLE INPUT ────────────────────────────────────────────────────────
input_data = {
    "input": "Can India attack Pakistan?",
    "output": "",
    "confidence": 0,
    "retries": 0, # Initialize retries
    "human_review": False # Initialize human_review flag
}

# ─── RUN ─────────────────────────────────────────────────────────────────
result = graph.invoke(input_data)

# ─── FINAL OUTPUT ────────────────────────────────────────────────────────
print("\n---")
print("\n✅ FINAL OUTPUT")
print("Answer:", result["output"])
print("Confidence:", result["confidence"])
print("Retries:", result["retries"])
print("Human Review Needed:", result["human_review"])