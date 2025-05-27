from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json

google_api_key = "AIzaSyBXT79IZQqRtnaYUApAtpW4epjKah1k2G4" # Replace with your actual API key
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

base_dir = "Knowledge_Base"
knowledge_list = []

def extract_knowledge(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.endswith(".json"):
            try:
                with open(item_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "topic" in data and "official_narrative" in data and "key_points" in data:
                        knowledge_list.append({
                            "topic": data["topic"],
                            "official_narrative": data["official_narrative"],
                            "key_points": data["key_points"]
                        })
            except Exception as e:
                print(f"Error reading file {item_path}: {e}")
        elif os.path.isdir(item_path):
            extract_knowledge(item_path)

extract_knowledge(base_dir)

# Print the extracted knowledge_list
print("Extracted knowledge_list:")
for item in knowledge_list:
    print(f"  Topic: {item['topic']}")
    print(f"  Official Narrative: {item['official_narrative']}")
    print(f"  Key Points: {item['key_points']}")
    print("-" * 20)

docs = [
    Document(page_content=item["official_narrative"], metadata={"topic": item["topic"], "key_points": ", ".join(item["key_points"])})
    for item in knowledge_list
]


# Initialize the vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Define metadata field information for self-querying
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

document_content_description = "Knowledge Base offcial narratives and facts"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

def get_relevant_info(query, k=50):
    results = retriever.get_relevant_documents(query, k=k)
    unique_relevant_info = []
    seen_content = set()

    if results:
        for doc in results:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_relevant_info.append({
                    "official_narrative": content,
                    "key_points": doc.metadata.get("key_points", "").split(", ") if doc.metadata.get("key_points") else []
                })
        return unique_relevant_info
    else:
        return "No information"

# Example queries
query1 = "Tell me about Pakistan"
answer1 = get_relevant_info(query1)
print(f"\nQuery: {query1}")
if isinstance(answer1, list):
    for item in answer1:
        print(f"  Official Narrative: {item['official_narrative']}")
        print(f"  Key Points: {item['key_points']}")
        print("-" * 20)
else:
    print(f"Answer: {answer1}\n")