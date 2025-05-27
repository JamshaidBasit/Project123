import operator
from typing_extensions import TypedDict, Literal
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
import time
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import List, Annotated, Dict, Callable
import json
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3

# Replace with your actual Groq API key
groq_api_key = "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"
google_api_key = "AIzaSyDXJgzKHH1vqwMAu0mRyIPwDmM8MIM7t60" # Replace with your actual API key
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# ─── LLM INIT ──────────────────────────────────────────────────────
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)

class State(TypedDict):
    report_text: str
    final_decision_report: str
    aggregate: Annotated[List[str], operator.add]
    main_node_output: Annotated[Dict, lambda a, b: {**a, **b}]
    metadata: Dict # To hold page, paragraph, title

# Define a type for agent functions
Agent = Callable[[State], Dict]

# Define a dictionary to hold the agents
available_agents: Dict[str, Agent] = {}

# Define the base directory for agents
AGENTS_FOLDER = "Agents"
base_dir = "Knowledge_Base_New"
knowledge_list = []

# --- Knowledge Base Extraction and Vector Store Initialization ---

def extract_knowledge(directory):
    """
    Recursively extracts knowledge from JSON files within the specified directory,
    including sensitive_aspects, recommended_terminology, and authoritative_sources.
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.endswith(".json"):
            try:
                with open(item_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Ensure essential fields are present
                    if all(k in data for k in ["topic", "official_narrative", "key_points"]):
                        knowledge_item = {
                            "topic": data["topic"],
                            "official_narrative": data["official_narrative"],
                            "key_points": data["key_points"],
                            "sensitive_aspects": data.get("sensitive_aspects", []),
                            "recommended_terminology": data.get("recommended_terminology", {}),
                            "authoritative_sources": data.get("authoritative_sources", [])
                        }
                        knowledge_list.append(knowledge_item)
            except Exception as e:
                print(f"Error reading file {item_path}: {e}")
        elif os.path.isdir(item_path):
            extract_knowledge(item_path)

# Extract knowledge from your base directory
extract_knowledge(base_dir)

# Print the extracted knowledge_list for verification
print("Extracted knowledge_list (full data):")
for item in knowledge_list:
    print(f"  Topic: {item['topic']}")
    print(f"  Official Narrative: {item['official_narrative']}")
    print(f"  Key Points: {item['key_points']}")
    print(f"  Sensitive Aspects: {item['sensitive_aspects']}")
    print(f"  Recommended Terminology: {item['recommended_terminology']}")
    print(f"  Authoritative Sources: {item['authoritative_sources']}")
    print("-" * 20)

# Create Langchain Document objects with only 'topic' and 'key_points' in metadata
docs = [
    Document(
        page_content=item["official_narrative"],
        metadata={
            "topic": item["topic"],
            "key_points": ", ".join(item["key_points"])
        }
    )
    for item in knowledge_list
]

# Initialize the vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Define metadata field information for self-querying (only topic and key_points)
metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="The topic of the knowledge document (string)",
        type="string",
    ),
    AttributeInfo(
        name="key_points",
        description="Key points related to the topic (comma-separated string)",
        type="string",
    ),
]

document_content_description = "Knowledge Base official narratives and facts"
llm_google = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key) # Renamed to avoid conflict with Groq LLM

retriever = SelfQueryRetriever.from_llm(
    llm_google, # Use the Google LLM for retrieval
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True # Set to True for debugging self-querying
)

def get_relevant_info(query, k=50):
    """
    Retrieves relevant documents based on the query and extracts specific fields.
    It will now merge with the full 'knowledge_list' to get all data.
    """
    results = retriever.get_relevant_documents(query, k=k)
    unique_relevant_info = []
    seen_content = set()

    if results:
        for doc in results:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                # Find the full knowledge item from the knowledge_list based on official_narrative
                full_item = next((item for item in knowledge_list if item["official_narrative"] == content), None)

                if full_item:
                    unique_relevant_info.append({
                        "official_narrative": full_item["official_narrative"],
                        "topic": full_item["topic"],
                        "key_points": full_item["key_points"],
                        "sensitive_aspects": full_item["sensitive_aspects"],
                        "recommended_terminology": full_item["recommended_terminology"],
                        "authoritative_sources": full_item["authoritative_sources"]
                    })
        return unique_relevant_info
    else:
        return "No information found for the query."


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
- Use the official policy guidelines and Knowledge Base to inform your judgment

Knowledge Base:
Official Narrative: {official_narrative}
Key Points: {key_points}
Sensitive Aspects: {sensitive_aspects}
Recommended Terminology: {recommended_terminology}
Authoritative Sources: {authoritative_sources}

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

# Function to register new agents
def register_agent(name: str, agent_function: Agent):
    available_agents[name] = agent_function

def create_review_agent(review_name: str, criteria: str, guidelines: str):
    prompt_template = PromptTemplate.from_template(TEMPLATE)
    def reviewer_agent(state: State):
        print(f"{review_name} agent called")
        report_text = state["report_text"]
        metadata = state["metadata"]
        # Retrieve relevant knowledge based on the report_text (query)
        relevant_knowledge = get_relevant_info(report_text)

        official_narrative = "No relevant official narrative found."
        key_points_str = "No relevant key points found."
        sensitive_aspects_str = "No relevant sensitive aspects found."
        recommended_terminology_str = "No relevant recommended terminology found."
        authoritative_sources_str = "No relevant authoritative sources found."

        if isinstance(relevant_knowledge, list) and relevant_knowledge:
            # Assuming the first result is the most relevant
            relevant_item = relevant_knowledge[0]
            official_narrative = relevant_item.get("official_narrative", official_narrative)
            key_points_str = ", ".join(relevant_item.get("key_points", []))
            sensitive_aspects_str = json.dumps(relevant_item.get("sensitive_aspects", [])) # Ensure JSON string
            recommended_terminology_str = json.dumps(relevant_item.get("recommended_terminology", {})) # Ensure JSON string
            authoritative_sources_str = ", ".join(relevant_item.get("authoritative_sources", []))

        prompt = prompt_template.format(
            text=report_text,
            page=metadata.get("page", "N/A"),
            paragraph=metadata.get("paragraph", "N/A"),
            title=metadata.get("title", "N/A"),
            specific_criteria=criteria,
            policy_guidelines=guidelines,
            official_narrative=official_narrative,
            key_points=key_points_str,
            sensitive_aspects=sensitive_aspects_str, # New field in prompt
            recommended_terminology=recommended_terminology_str, # New field in prompt
            authoritative_sources=authoritative_sources_str # New field in prompt
        )
        print(prompt) # Uncomment for debugging the full prompt
        response = llm.invoke(prompt)
        try:
            output = json.loads(response.content)
            print(f"{review_name} Output (JSON):\n{output}")
            return {review_name: output, "aggregate": [f"{review_name} Output: {output}"], "main_node_output": {review_name: output}}
        except json.JSONDecodeError:
            error_message = f"Error decoding JSON response from {review_name}: {response.content}"
            print(error_message)
            return {review_name: {"error": error_message}, "aggregate": [f"{review_name} Output: Error"], "main_node_output": {review_name: {"error": error_message}}}
    return reviewer_agent

# Function to load and register agents from the Agents folder
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

# Load agents from the Agents folder
load_agents()

def main_node(state: State):
    print("main_node called")
    return {}

def final_report_generator(state: State):
    print("final_report_generator agent called")
    report_parts = {}
    for agent_name, output_data in state.items():
        # Check if the key corresponds to an agent that was executed and has an output
        if agent_name in available_agents and isinstance(output_data, dict) and agent_name in output_data:
            report_parts[agent_name] = output_data[agent_name]

    aggregate_history = state.get("aggregate", [])

    final_decision_report = "*Review Report:*\n\n"
    for name, result in report_parts.items():
        final_decision_report += f"*{name}*: {result}\n\n"
    final_decision_report += f"*Aggregate History:*\n{chr(10).join(aggregate_history)}\n\nThis report compiles insights from various critical reviews."
    return {"final_decision_report": final_decision_report}

graph_builder = StateGraph(State)

graph_builder.add_node("main_node", main_node)
graph_builder.add_node("fnl_rprt", final_report_generator)

graph_builder.add_edge(START, "main_node")

# Dynamically add edges for each available agent loaded from folders
for agent_name in available_agents:
    graph_builder.add_node(agent_name, available_agents[agent_name])
    graph_builder.add_edge("main_node", agent_name)
    graph_builder.add_edge(agent_name, "fnl_rprt")

graph_builder.add_edge("fnl_rprt", END)

graph = graph_builder.compile()


# --- Database Interaction and Graph Invocation ---

# Connect to the SQLite database
conn = sqlite3.connect("test0001.db")
cursor = conn.cursor()

# Get all distinct pages
cursor.execute("SELECT DISTINCT page_number FROM pdf_chunks ORDER BY page_number")
pages = [row[0] for row in cursor.fetchall()]

# Dictionary to store merged text for each chunk ID
merged_texts = {}

for page_num in pages:
    # Fetch all chunks for the current page with their IDs
    cursor.execute("""
        SELECT id, chunk_number, chunk_text
        FROM pdf_chunks
        WHERE page_number = ?
        ORDER BY chunk_number
    """, (page_num,))
    chunks_with_id = cursor.fetchall()

    for i in range(len(chunks_with_id)):
        current_id = chunks_with_id[i][0]
        surrounding_chunks = chunks_with_id[max(i - 2, 0):min(i + 3, len(chunks_with_id))]
        merged_text = " ".join([chunk[2] for chunk in surrounding_chunks])
        merged_texts[current_id] = merged_text

# Retrieve the first three rows from the pdf_chunks table
cursor.execute("SELECT id, book_title, page_number, chunk_number, chunk_text FROM pdf_chunks LIMIT 1")
first_three_rows = cursor.fetchall()

# Close the database connection
conn.close()

if first_three_rows:
    print("--- Executing graph.invoke() with merged text for the first database entry ---")
    for row in first_three_rows:
        row_id, book_title, page_number, chunk_number, original_chunk_text = row
        # Get the merged text using the ID
        merged_text_for_id = merged_texts.get(row_id, original_chunk_text) # Fallback to original if ID not found
        print(f"Processing chunk ID: {row_id}, Page: {page_number}, Paragraph: {chunk_number}")
        print(f"Chunk Text: {merged_text_for_id}\n")

        report_data = {
            "report_text": merged_text_for_id,
            "metadata": {"page": page_number, "paragraph": chunk_number, "title": book_title}
        }
        print(f"\n--- Processing ID: {row_id} ---")
        # Invoke the graph with the current report_data (now using merged text)
        result_with_review = graph.invoke(report_data)
        print("--- Result with all review agents ---")
        print(result_with_review)
else:
    print("No data found in the 'pdf_chunks' table.")