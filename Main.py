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
# It's highly recommended to use environment variables for API keys in production
groq_api_key = "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"
google_api_key = "AIzaSyBm0uGIbuxUyFCqKTC1_mawa6rxA9hKzNw"  # Replace with your actual API key
# Initialize Google Generative AI Embeddings for vector store creation
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# ─── LLM INITIALIZATION ──────────────────────────────────────────────────────
# Initialize the ChatGroq language model for review agents
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)

# ─── STATE DEFINITION ────────────────────────────────────────────────────────
# Define the state schema for the Langgraph workflow
class State(TypedDict):
    """
    Represents the state of the Langgraph workflow.

    Attributes:
        report_text (str): The main text segment being analyzed.
        final_decision_report (str): The compiled final report of all reviews.
        aggregate (Annotated[List[str], operator.add]): A list to accumulate
                                                        messages/outputs from agents.
        main_node_output (Annotated[Dict, lambda a, b: {**a, **b}]): Output
                                                                    from the main node.
        metadata (Dict): Contains contextual information about the report_text
                         (e.g., page, paragraph, title).
    """
    report_text: str
    final_decision_report: str     
    aggregate: Annotated[List[str], operator.add]
    main_node_output: Annotated[Dict, lambda a, b: {**a, **b}]
    metadata: Dict # To hold page, paragraph, title

# Define a type for agent functions for clear type hinting
Agent = Callable[[State], Dict]

# Define a dictionary to hold all available agents, mapping names to their functions
available_agents: Dict[str, Agent] = {}

# --- KNOWLEDGE BASE EXTRACTION AND VECTOR STORE INITIALIZATION ────────────────
# Define the path to the SQLite database containing the knowledge base
KNOWLEDGE_BASE_DB_PATH = 'knowledge_base_flat001.db' # Ensure this path is correct

# Initialize an empty list to store extracted knowledge
knowledge_list = []

def extract_knowledge_from_db(db_path: str) -> List[Dict]:
    """
    Extracts knowledge entries from a given SQLite database.

    Each entry is expected to have 'topic' and 'json_data' columns, where
    'json_data' is a JSON string containing official_narrative, key_points,
    sensitive_aspects, recommended_terminology, and authoritative_sources.

    Args:
        db_path (str): The file path to the SQLite database.

    Returns:
        List[Dict]: A list of dictionaries, each representing a knowledge item
                    with parsed fields.
    """
    conn = None
    extracted_knowledge = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT topic, json_data FROM kb_data")
        rows = cursor.fetchall()

        for topic, json_data_str in rows:
            try:
                data = json.loads(json_data_str)
                # Ensure essential fields are present in the JSON data
                if all(k in data for k in ["topic", "official_narrative", "key_points"]):
                    knowledge_item = {
                        "topic": data["topic"],
                        "official_narrative": data["official_narrative"],
                        "key_points": data["key_points"],
                        "sensitive_aspects": data.get("sensitive_aspects", []),
                        "recommended_terminology": data.get("recommended_terminology", {}),
                        "authoritative_sources": data.get("authoritative_sources", [])
                    }
                    extracted_knowledge.append(knowledge_item)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for topic '{topic}': {e}")
            except Exception as e:
                print(f"Error processing topic '{topic}': {e}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
    return extracted_knowledge

# Extract knowledge from the specified database path
knowledge_list = extract_knowledge_from_db(KNOWLEDGE_BASE_DB_PATH)

# Print the extracted knowledge_list for verification (optional, can be removed in production)
print("Extracted knowledge_list (full data):")
for item in knowledge_list:
    print(f"   Topic: {item['topic']}")
    print(f"   Official Narrative: {item['official_narrative']}")
    print(f"   Key Points: {item['key_points']}")
    print(f"   Sensitive Aspects: {item['sensitive_aspects']}")
    print(f"   Recommended Terminology: {item['recommended_terminology']}")
    print(f"   Authoritative Sources: {item['authoritative_sources']}")
    print("-" * 20)

# Create Langchain Document objects from the extracted knowledge.
# Only 'topic' and 'key_points' are included in metadata for specific retrieval needs.
docs = [
    Document(
        page_content=item["official_narrative"],
        metadata={
            "topic": item["topic"],
            "key_points": ", ".join(item["key_points"]) # Join key points into a single string
        }
    )
    for item in knowledge_list
]

# Initialize the Chroma vector store with the created documents and embeddings
vectorstore = Chroma.from_documents(docs, embeddings)

# Define metadata field information for self-querying.
# This tells the SelfQueryRetriever which metadata fields are available for filtering.
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

# Description of the document content for the self-query retriever
document_content_description = "Knowledge Base official narratives and facts"

# Initialize a separate Google LLM for self-querying to avoid conflicts with Groq LLM
llm_google = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

# Initialize the SelfQueryRetriever, enabling it to construct queries over the vector store's metadata
retriever = SelfQueryRetriever.from_llm(
    llm_google, # Use the Google LLM for generating queries
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True # Set to True for debugging the self-querying process
)

def get_relevant_info(query: str, k: int = 50) -> List[Dict]:
    """
    Retrieves relevant documents from the vector store based on a query
    and merges them with the full knowledge base data.

    Args:
        query (str): The user query or text to find relevant information for.
        k (int): The number of relevant documents to retrieve.

    Returns:
        List[Dict]: A list of dictionaries, each containing the full details
                    of relevant knowledge items. Returns a string message if no
                    information is found.
    """
    results = retriever.get_relevant_documents(query, k=k)
    unique_relevant_info = []
    seen_content = set()

    if results:
        for doc in results:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                # Find the full knowledge item from the knowledge_list based on the 'official_narrative'
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

# ─── PROMPT TEMPLATE ────────────────────────────────────────────────────────
# Define the base prompt template for the review agents
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

# ─── AGENT CREATION AND REGISTRATION ────────────────────────────────────────
def register_agent(name: str, agent_function: Agent):
    """
    Registers an agent function under a given name in the global available_agents dictionary.

    Args:
        name (str): The unique name for the agent.
        agent_function (Agent): The callable agent function.
    """
    available_agents[name] = agent_function

def create_review_agent(review_name: str, criteria: str, guidelines: str) -> Agent:
    """
    Creates a specialized review agent function based on specific criteria and guidelines.

    Args:
        review_name (str): The name of the review agent.
        criteria (str): Specific criteria for this agent to evaluate the text against.
        guidelines (str): Official policy guidelines for this agent to follow.

    Returns:
        Agent: A callable agent function that takes a State and returns a Dict.
    """
    prompt_template = PromptTemplate.from_template(TEMPLATE)
    def reviewer_agent(state: State) -> Dict:
        print(f"{review_name} agent called")
        report_text = state["report_text"]
        metadata = state["metadata"]

        # Retrieve relevant knowledge based on the current report text
        relevant_knowledge = get_relevant_info(report_text)

        # Initialize knowledge base fields with default "No information found." messages
        official_narrative = "No relevant official narrative found."
        key_points_str = "No relevant key points found."
        sensitive_aspects_str = "No relevant sensitive aspects found."
        recommended_terminology_str = "No relevant recommended terminology found."
        authoritative_sources_str = "No relevant authoritative sources found."

        # If relevant knowledge is found, populate the fields for the prompt
        if isinstance(relevant_knowledge, list) and relevant_knowledge:
            # Assuming the first result is the most relevant for this example
            relevant_item = relevant_knowledge[0]
            official_narrative = relevant_item.get("official_narrative", official_narrative)
            key_points_str = ", ".join(relevant_item.get("key_points", []))
            # Ensure JSON strings for dictionary/list fields
            sensitive_aspects_str = json.dumps(relevant_item.get("sensitive_aspects", []))
            recommended_terminology_str = json.dumps(relevant_item.get("recommended_terminology", {}))
            authoritative_sources_str = ", ".join(relevant_item.get("authoritative_sources", []))

        # Format the prompt with the current text, metadata, and retrieved knowledge
        prompt = prompt_template.format(
            text=report_text,
            page=metadata.get("page", "N/A"),
            paragraph=metadata.get("paragraph", "N/A"),
            title=metadata.get("title", "N/A"),
            specific_criteria=criteria,
            policy_guidelines=guidelines,
            official_narrative=official_narrative,
            key_points=key_points_str,
            sensitive_aspects=sensitive_aspects_str,
            recommended_terminology=recommended_terminology_str,
            authoritative_sources=authoritative_sources_str
        )
        print(prompt) # Uncomment for debugging the full prompt sent to LLM

        # Invoke the LLM with the constructed prompt
        response = llm.invoke(prompt)
        try:
            # Attempt to parse the LLM's response as JSON
            output = json.loads(response.content)
            print(f"{review_name} Output (JSON):\n{output}")
            # Return the agent's output and update the aggregate and main_node_output
            return {review_name: output, "aggregate": [f"{review_name} Output: {output}"], "main_node_output": {review_name: output}}
        except json.JSONDecodeError:
            # Handle cases where the LLM response is not valid JSON
            error_message = f"Error decoding JSON response from {review_name}: {response.content}"
            print(error_message)
            return {review_name: {"error": error_message}, "aggregate": [f"{review_name} Output: Error"], "main_node_output": {review_name: {"error": error_message}}}
    return reviewer_agent

# ─── AGENT LOADING FROM DATABASE ─────────────────────────────────────────────
def load_agents_from_db(db_path: str):
    """
    Loads agent configurations (name, criteria, guidelines) from a SQLite database
    and registers them as review agents.

    Args:
        db_path (str): The file path to the SQLite database containing agent configurations.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Fetch agent names and their JSON configurations
        cursor.execute('SELECT agent_name, json_content FROM json_data')
        rows = cursor.fetchall()

        for agent_name, json_content in rows:
            try:
                content = json.loads(json_content)
                criteria = content.get("criteria")
                guidelines = content.get("guidelines")

                if agent_name and criteria and guidelines:
                    # Create and register the agent if all required fields are present
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

# ─── CORE WORKFLOW NODES ─────────────────────────────────────────────────────
def main_node(state: State) -> Dict:
    """
    The initial node in the graph. It typically receives the initial state
    and can perform any setup or initial processing before routing to other agents.

    Args:
        state (State): The current state of the workflow.

    Returns:
        Dict: An empty dictionary, indicating no direct state modification from this node.
    """
    print("main_node called")
    return {}

def final_report_generator(state: State) -> Dict:
    """
    Aggregates the outputs from all review agents and generates a comprehensive
    final decision report.

    Args:
        state (State): The current state containing outputs from all agents.

    Returns:
        Dict: A dictionary containing the generated final decision report.
    """
    print("final_report_generator agent called")
    report_parts = {}

    # Collect results from all available agents in the state
    for agent_name in available_agents:
        result = state.get(agent_name, {})
        report_parts[agent_name] = result

    # Get the aggregate history from the state
    aggregate_history = state.get("aggregate", [])
    final_decision_report = "**Review Report:**\n\n"

    # Append each agent's result to the final report
    for name, result in report_parts.items():
        final_decision_report += f"**{name}**: {result}\n\n"

    # Append the aggregated history
    final_decision_report += (
        "**Aggregate History:**\n" +
        "\n".join(aggregate_history) +
        "\n\nThis report compiles insights from various critical reviews."
    )

    return {"final_decision_report": final_decision_report}

# ─── GRAPH BUILDING ──────────────────────────────────────────────────────────
# Define the path to the database containing agent configurations
DATABASE_PATH = r"C:\Users\USER\Desktop\Mcs_Project\MCS_Project\reviews_database0.db"
# Load agents dynamically from the database
load_agents_from_db(DATABASE_PATH)

# Initialize the StateGraph with the defined State
graph_builder = StateGraph(State)

# Add the main_node and final_report_generator nodes to the graph
graph_builder.add_node("main_node", main_node)
graph_builder.add_node("fnl_rprt", final_report_generator)

# Set the entry point of the graph to "main_node"
graph_builder.add_edge(START, "main_node")

# Dynamically add edges from "main_node" to each loaded agent, and then from each agent to the "fnl_rprt"
for agent_name in available_agents:
    graph_builder.add_node(agent_name, available_agents[agent_name]) # Add each agent as a node
    graph_builder.add_edge("main_node", agent_name) # Edge from main_node to the agent
    graph_builder.add_edge(agent_name, "fnl_rprt") # Edge from the agent to the final report generator

# Set the exit point of the graph to "fnl_rprt"
graph_builder.add_edge("fnl_rprt", END)

# Compile the graph for execution
graph = graph_builder.compile()

#=======================================================================================================================================================================

# Connect to the SQLite database containing PDF chunks
conn = sqlite3.connect(r"C:\Users\USER\Desktop\Mcs_Project\MCS_Project\test0001.db")
cursor = conn.cursor()

# Get all distinct page numbers from the pdf_chunks table, ordered numerically
cursor.execute("SELECT DISTINCT page_number FROM pdf_chunks ORDER BY page_number")
pages = [row[0] for row in cursor.fetchall()]

# Dictionary to store merged text for each chunk ID. This approach aims to provide
# more context to each chunk by merging it with surrounding chunks.
merged_texts = {}

# Iterate through each page to merge chunks
for page_num in pages:
    # Fetch all chunks for the current page, including their IDs and text, ordered by chunk number
    cursor.execute("""
        SELECT id, chunk_number, chunk_text
        FROM pdf_chunks
        WHERE page_number = ?
        ORDER BY chunk_number
    """, (page_num,))
    chunks_with_id = cursor.fetchall()

    # For each chunk, merge it with its 2 preceding and 2 succeeding chunks (if available)
    for i in range(len(chunks_with_id)):
        current_id = chunks_with_id[i][0]
        # Determine the slice for surrounding chunks to include context
        surrounding_chunks = chunks_with_id[max(i - 2, 0):min(i + 3, len(chunks_with_id))]
        # Join the text of the surrounding chunks to create a merged text
        merged_text = " ".join([chunk[2] for chunk in surrounding_chunks])
        merged_texts[current_id] = merged_text

# Retrieve the first row from the pdf_chunks table to demonstrate the workflow.
# This can be modified to iterate through all chunks or a specific subset.
cursor.execute("SELECT id, book_title, page_number, chunk_number, chunk_text FROM pdf_chunks LIMIT 1")
first_row = cursor.fetchall()

# Close the database connection
conn.close()

# Execute the graph for each retrieved row (currently only the first one)
if first_row:
    print("--- Executing graph.invoke() with merged text for the first database entry ---")
    for row in first_row:
        row_id, book_title, page_number, chunk_number, original_chunk_text = row
        # Get the merged text for the current chunk ID, falling back to original if not found
        merged_text_for_id = merged_texts.get(row_id, original_chunk_text)
        print(f"Processing chunk ID: {row_id}, Page: {page_number}, Paragraph: {chunk_number}")
        print(f"Chunk Text: {merged_text_for_id}\n")

        # Prepare the input state for the Langgraph workflow
        report_data = {
            "report_text": merged_text_for_id,
            "metadata": {"page": page_number, "paragraph": chunk_number, "title": book_title}
        }
        print(f"\n--- Processing ID: {row_id} ---")
        # Invoke the graph with the current report data (using merged text)
        result_with_review = graph.invoke(report_data)
        print("--- Result with all review agents ---")
        print(result_with_review)
else:
    print("No data found in the 'pdf_chunks' table.")


    

##report_text\
##final_decision_report
##aggregate
##main node output
##meta data