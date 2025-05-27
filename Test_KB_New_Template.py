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

base_dir = "Knowledge_Base_New"
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
                # Find the full knowledge item from the knowledge_list
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