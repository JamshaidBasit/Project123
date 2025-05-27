import os
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

# Make sure to replace with your actual API key
google_api_key = "AIzaSyBXT79IZQqRtnaYUApAtpW4epjKah1k2G4"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

base_dir = "Knowledge_Base"
knowledge_list = []

def extract_knowledge(directory):
    """
    Recursively extracts knowledge from JSON files within the specified directory,
    now including sensitive_aspects, recommended_terminology, and authoritative_sources.
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
                            "sensitive_aspects": data.get("sensitive_aspects", []), # New field
                            "recommended_terminology": data.get("recommended_terminology", {}), # New field
                            "authoritative_sources": data.get("authoritative_sources", []) # New field
                        }
                        knowledge_list.append(knowledge_item)
            except Exception as e:
                print(f"Error reading file {item_path}: {e}")
        elif os.path.isdir(item_path):
            extract_knowledge(item_path)

# Extract knowledge from your base directory
extract_knowledge(base_dir)

# Print the extracted knowledge_list for verification
print("Extracted knowledge_list:")
for item in knowledge_list:
    print(f"  Topic: {item['topic']}")
    print(f"  Official Narrative: {item['official_narrative']}")
    print(f"  Key Points: {item['key_points']}")
    print(f"  Sensitive Aspects: {item['sensitive_aspects']}")
    print(f"  Recommended Terminology: {item['recommended_terminology']}")
    print(f"  Authoritative Sources: {item['authoritative_sources']}")
    print("-" * 20)

# Create Langchain Document objects, now including the new metadata fields
docs = []
for item in knowledge_list:
    # Convert lists/dictionaries to strings for metadata if they are not directly supported
    sensitive_aspects_str = json.dumps(item["sensitive_aspects"])
    recommended_terminology_str = json.dumps(item["recommended_terminology"])
    authoritative_sources_str = ", ".join(item["authoritative_sources"])

    docs.append(
        Document(
            page_content=item["official_narrative"],
            metadata={
                "topic": item["topic"],
                "key_points": ", ".join(item["key_points"]),
                "sensitive_aspects": sensitive_aspects_str, # New metadata field
                "recommended_terminology": recommended_terminology_str, # New metadata field
                "authoritative_sources": authoritative_sources_str # New metadata field
            }
        )
    )

# Initialize the vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Define metadata field information for self-querying, including the new fields
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
    AttributeInfo(
        name="sensitive_aspects",
        description="A JSON string representing a list of sensitive aspects related to the topic, each with a 'topic', 'approved_framing', and 'problematic_framing'.",
        type="string", # Stored as string, will need parsing if used in logic
    ),
    AttributeInfo(
        name="recommended_terminology",
        description="A JSON string representing preferred and avoided terminology for the topic.",
        type="string", # Stored as string, will need parsing if used in logic
    ),
    AttributeInfo(
        name="authoritative_sources",
        description="A comma-separated string of authoritative sources for the information.",
        type="string",
    ),
]

document_content_description = "Knowledge Base official narratives and facts about Pakistan"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True # Set to True for debugging self-querying
)

def get_relevant_info(query, k=50):
    """
    Retrieves relevant documents based on the query and extracts specific fields.
    """
    results = retriever.get_relevant_documents(query, k=k)
    unique_relevant_info = []
    seen_content = set()

    if results:
        for doc in results:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                # Parse sensitive_aspects and recommended_terminology back to Python objects
                try:
                    sensitive_aspects = json.loads(doc.metadata.get("sensitive_aspects", "[]"))
                except json.JSONDecodeError:
                    sensitive_aspects = []

                try:
                    recommended_terminology = json.loads(doc.metadata.get("recommended_terminology", "{}"))
                except json.JSONDecodeError:
                    recommended_terminology = {}

                unique_relevant_info.append({
                    "official_narrative": content,
                    "topic": doc.metadata.get("topic", "N/A"),
                    "key_points": doc.metadata.get("key_points", "").split(", ") if doc.metadata.get("key_points") else [],
                    "sensitive_aspects": sensitive_aspects,
                    "recommended_terminology": recommended_terminology,
                    "authoritative_sources": doc.metadata.get("authoritative_sources", "").split(", ") if doc.metadata.get("authoritative_sources") else []
                })
        return unique_relevant_info
    else:
        return "No information"

# Example queries
query1 = "Tell me about the 1971 war and its sensitive aspects."
answer1 = get_relevant_info(query1)
print(f"\nQuery: {query1}")
if isinstance(answer1, list):
    for item in answer1:
        print(f"  Topic: {item['topic']}")
        print(f"  Official Narrative: {item['official_narrative']}")
        print(f"  Key Points: {item['key_points']}")
        print(f"  Sensitive Aspects: {item['sensitive_aspects']}")
        print(f"  Recommended Terminology: {item['recommended_terminology']}")
        print(f"  Authoritative Sources: {item['authoritative_sources']}")
        print("-" * 20)
else:
    print(f"Answer: {answer1}\n")

query2 = "What are the economic challenges of Pakistan and what should I avoid saying?"
answer2 = get_relevant_info(query2)
print(f"\nQuery: {query2}")
if isinstance(answer2, list):
    for item in answer2:
        print(f"  Topic: {item['topic']}")
        print(f"  Official Narrative: {item['official_narrative']}")
        print(f"  Key Points: {item['key_points']}")
        print(f"  Sensitive Aspects: {item['sensitive_aspects']}")
        print(f"  Recommended Terminology: {item['recommended_terminology']}")
        print(f"  Authoritative Sources: {item['authoritative_sources']}")
        print("-" * 20)
else:
    print(f"Answer: {answer2}\n")

query3 = "What are the diplomatic relations between Pakistan and China, and who are the authoritative sources?"
answer3 = get_relevant_info(query3)
print(f"\nQuery: {query3}")
if isinstance(answer3, list):
    for item in answer3:
        print(f"  Topic: {item['topic']}")
        print(f"  Official Narrative: {item['official_narrative']}")
        print(f"  Key Points: {item['key_points']}")
        print(f"  Sensitive Aspects: {item['sensitive_aspects']}")
        print(f"  Recommended Terminology: {item['recommended_terminology']}")
        print(f"  Authoritative Sources: {item['authoritative_sources']}")
        print("-" * 20)
else:
    print(f"Answer: {answer3}\n")