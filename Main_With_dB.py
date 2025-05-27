import sqlite3
import json
import os
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
# AGENTS_FOLDER is no longer needed as agents are loaded from the database

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
        print(prompt)

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

# ─── AGENT LOADING FROM DATABASE ─────────────────────────────────────
def load_agents_from_db(db_path: str):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT agent_name, json_content FROM json_data')
        rows = cursor.fetchall()

        for agent_name, json_content in rows:
            try:
                content = json.loads(json_content)
                criteria = content.get("criteria")
                guidelines = content.get("guidelines")

                if agent_name and criteria and guidelines:
                    agent = create_review_agent(agent_name, criteria, guidelines)
                    register_agent(agent_name, agent)
                    print(f"Agent '{agent_name}' loaded from database.")
                else:
                    print(f"Error: Missing 'criteria' or 'guidelines' for agent '{agent_name}' in database.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON for agent '{agent_name}' from database.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

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
# Load agents from the new database
DATABASE_PATH = r"C:\Users\USER\Desktop\Mcs_Project\MCS_Project\reviews_database0.db"
load_agents_from_db(DATABASE_PATH)

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

#=======================================================================================================================================================================

import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect(r"C:\Users\USER\Desktop\Mcs_Project\MCS_Project\test0001.db")
cursor = conn.cursor()

# Get all distinct page numbers from the pdf_chunks table
cursor.execute("SELECT DISTINCT page_number FROM pdf_chunks ORDER BY page_number")
pages = [row[0] for row in cursor.fetchall()]

# Dictionary to store merged text for each chunk ID
merged_texts = {}


# Iterate through each page to fetch and merge surrounding chunks
for page_num in pages:
    cursor.execute("""
        SELECT id, chunk_number, chunk_text
        FROM pdf_chunks
        WHERE page_number = ?
        ORDER BY chunk_number
    """, (page_num,))
    
    chunks_with_id = cursor.fetchall()

    for i in range(len(chunks_with_id)):
        current_id = chunks_with_id[i][0]
        
        # Get chunks within a window of [-2, +2] around the current chunk
        surrounding_chunks = chunks_with_id[max(i - 2, 0):min(i + 3, len(chunks_with_id))]
        merged_text = " ".join([chunk[2] for chunk in surrounding_chunks])
        
        merged_texts[current_id] = merged_text

# Retrieve the first three rows from the pdf_chunks table
cursor.execute("SELECT id, book_title, page_number, chunk_number, chunk_text FROM pdf_chunks LIMIT 3")
first_three_rows = cursor.fetchall()

# Close the database connection
conn.close()

# If data exists, process each of the first three entries
if first_three_rows:
    print("--- Executing graph.invoke() with merged text for the first three database entries ---")
    
    for row in first_three_rows:
        row_id, book_title, page_number, chunk_number, original_chunk_text = row
        
        # Get the merged text using the ID (fallback to original if not found)
        merged_text_for_id = merged_texts.get(row_id, original_chunk_text)
        
        print(f"\n--- Processing ID: {row_id} ---")
        print(merged_text_for_id)
        
        report_data = {
            "report_text": merged_text_for_id,
            "metadata": {
                "page": page_number,
                "paragraph": chunk_number,
                "title": book_title
            }
        }

        # Invoke the graph with the current report_data
        result_with_review = graph.invoke(report_data)

        print("--- Result with all review agents ---")
        print(result_with_review)

else:
    print("No data found in the 'pdf_chunks' table.")