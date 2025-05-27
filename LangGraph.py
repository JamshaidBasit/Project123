import os
import json
import time
from typing import List, Annotated, Dict, Callable
from typing_extensions import TypedDict, Literal

from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from langchain.prompts import PromptTemplate  
from langchain_groq import ChatGroq

# ─── API KEY SETUP ──────────────────────────────────────────────
# Replace with your actual Groq API key
groq_api_key = "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"

# ─── LLM INIT ───────────────────────────────────────────────────
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    groq_api_key=groq_api_key
)

# ─── TYPE DEFINITIONS ───────────────────────────────────────────
class State(TypedDict):
    report_text: str
    final_decision_report: str 
    aggregate: Annotated[List[str], list.__add__]
    main_node_output: Annotated[Dict, lambda a, b: {**a, **b}]
    metadata: Dict  # To hold page, paragraph, title

Agent = Callable[[State], Dict]
available_agents: Dict[str, Agent] = {}  

# ─── CONFIG ─────────────────────────────────────────────────────
AGENTS_FOLDER = r"C:\Users\USER\Desktop\Mcs_Project\MCS_Project\Agents"

TEMPLATE = """
You are an expert alignment reviewer for books about Pakistan's history and politics.
Your task is to identify content that may not align with Pakistan's official policies, narratives, or national interests.

Current text to analyze:
{text}

Page: {page}
Paragraph: {paragraph}
Context: This text is from a book titled "{title}".

Your task is to determine if this text contains any content that:
{specific_criteria}

If you find any issues, provide:
1. The specific problematic text
2. A brief observation explaining why it's problematic
3. A recommendation (delete, rephrase, fact-check, or provide references)

Follow these guidelines:
- Be objective in your assessment
- Consider the national interest of Pakistan
- Focus only on real alignment issues, not stylistic concerns
- Use the official policy guidelines to inform your judgment

Official policy guidelines:
{policy_guidelines}

Respond in JSON format:
{{
  "issues_found": true/false,
  "problematic_text": "exact text that is problematic",
  "observation": "brief explanation of the issue",
  "recommendation": "delete/rephrase/fact-check/provide references"
}}
"""

# ─── AGENT REGISTRATION ─────────────────────────────────────────
def register_agent(name: str, agent_function: Agent):
    available_agents[name] = agent_function

def create_review_agent(review_name: str, criteria: str, guidelines: str) -> Agent:
    prompt_template = PromptTemplate.from_template(TEMPLATE)

    def reviewer_agent(state: State) -> Dict:
        print(f"{review_name} agent called")
        report_text = state["report_text"]
        metadata = state["metadata"]

        prompt = prompt_template.format(
            text=report_text,
            page=metadata.get("page", "N/A"),
            paragraph=metadata.get("paragraph", "N/A"),
            title=metadata.get("title", "N/A"),
            specific_criteria=criteria,
            policy_guidelines=guidelines
        )

        response = llm.invoke(prompt)

        try:
            output = json.loads(response.content)
            print(f"{review_name} Output (JSON):\n{output}")
            return {
                review_name: output,
                "aggregate": [f"{review_name} Output: {output}"],
                "main_node_output": {review_name: output}
            }
        except json.JSONDecodeError:
            error_message = f"Error decoding JSON response from {review_name}: {response.content}"
            print(error_message)
            return {
                review_name: {"error": error_message},
                "aggregate": [f"{review_name} Output: Error"],
                "main_node_output": {review_name: {"error": error_message}}
            }

    return reviewer_agent

# ─── AGENT LOADING FROM DISK ─────────────────────────────────────
def load_agents():
    if not os.path.exists(AGENTS_FOLDER):
        print(f"Warning: Agents folder '{AGENTS_FOLDER}' not found.")
        return

    for item_name in os.listdir(AGENTS_FOLDER):
        agent_path = os.path.join(AGENTS_FOLDER, item_name)

        if os.path.isdir(agent_path):
            config_file = os.path.join(agent_path, "config.json")

            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)

                    agent_name = config.get("name")
                    criteria = config.get("criteria")
                    guidelines = config.get("guidelines")

                    if agent_name and criteria and guidelines:
                        agent = create_review_agent(agent_name, criteria, guidelines)
                        register_agent(agent_name, agent)
                        print(f"Agent '{agent_name}' loaded from '{agent_path}'.")
                    else:
                        print(f"Error: Missing 'name', 'criteria', or 'guidelines' in '{config_file}'.")
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from '{config_file}'.")
                except FileNotFoundError:
                    print(f"Error: Configuration file not found at '{config_file}'.")
            else:
                print(f"Warning: No 'config.json' found in '{agent_path}'.")

# ─── CORE WORKFLOW NODES ─────────────────────────────────────────
def main_node(state: State) -> Dict:
    print("main_node called")
    return {}

def final_report_generator(state: State) -> Dict:
    print("final_report_generator agent called")
    report_parts = {}

    for agent_name in available_agents:
        result = state.get(agent_name, {})
        report_parts[agent_name] = result

    aggregate_history = state.get("aggregate", [])
    final_decision_report = "**Review Report:**\n\n"

    for name, result in report_parts.items():
        final_decision_report += f"**{name}**: {result}\n\n"

    final_decision_report += (
        "**Aggregate History:**\n" +
        "\n".join(aggregate_history) +
        "\n\nThis report compiles insights from various critical reviews."
    )

    return {"final_decision_report": final_decision_report}

# ─── GRAPH BUILDING ──────────────────────────────────────────────
load_agents()

graph_builder = StateGraph(State)

graph_builder.add_node("main_node", main_node)
graph_builder.add_node("fnl_rprt", final_report_generator)

graph_builder.add_edge(START, "main_node")

for agent_name in available_agents:
    graph_builder.add_node(agent_name, available_agents[agent_name])
    graph_builder.add_edge("main_node", agent_name)
    graph_builder.add_edge(agent_name, "fnl_rprt")

graph_builder.add_edge("fnl_rprt", END)

graph = graph_builder.compile()
