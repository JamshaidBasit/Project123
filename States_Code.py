import json
import time
import os

def analyze_text_chunk_agent(text_chunk, agent_id, paragraph_id, chunk_index):
    """Simulates the analysis of a text chunk by a specific agent."""
    print(f"Agent {agent_id}: Analyzing Paragraph {paragraph_id}, Chunk {chunk_index}: '{text_chunk[:20]}...'")
    time.sleep(1)
    # Simulate potential errors during analysis
    if text_chunk.startswith("Error"):
        raise ValueError(f"Simulated error during analysis by agent {agent_id} on chunk {chunk_index}")
    return {"agent_id": agent_id, "paragraph_id": paragraph_id, "chunk_index": chunk_index, "result": f"Processed by agent {agent_id}"}

def save_agents_state(agents_state):
    """Saves the current state of all agents to a file.""" 
    state_file = "agents_state.json"           
    try:
        with open(state_file, "w") as f:
            json.dump(agents_state, f)
        print("Agents' state saved.")
    except IOError as e: 
        print(f"Error saving agents' state: {e}")

def load_agents_state():  
    """Loads the previous state of all agents from a file."""
    state_file = "agents_state.json"
    if not os.path.exists(state_file):
        print(f"State file '{state_file}' no t found. Creating an empty one.")
        return {}
    else:   
        try:
            with open(state_file, "r") as f:
                agents_state = json.load(f)  
            print("Agents' state loaded.")
            return agents_state
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{state_file}'. Starting with an empty state. Error: {e}")
            return {}
        except IOError as e:
            print(f"Error reading agents' state file: {e}. Starting with an empty state.")
            return {}

def save_partial_results(paragraph_id, results):
    """Saves the partial analysis results for a paragraph to a temporary JSON file."""
    filename = f"{paragraph_id}_partial_results.json"
    try:
        with open(filename, "w") as f:
            json.dump(results, f)
    except IOError as e:
        print(f"Error saving partial results for {paragraph_id}: {e}")

def load_partial_results(paragraph_id):
    """Loads the partial analysis results for a paragraph from a temporary JSON file."""
    filename = f"{paragraph_id}_partial_results.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding partial results for {paragraph_id}: {e}. Starting with empty partial results.")
            return []
        except IOError as e:
            print(f"Error reading partial results for {paragraph_id}: {e}. Starting with empty partial results.")
            return []
    return []

def delete_partial_results(paragraph_id):
    """Deletes the partial analysis results file."""
    filename = f"{paragraph_id}_partial_results.json"
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except OSError as e:
        print(f"Error deleting partial results file for {paragraph_id}: {e}")

def reset_agents_state():
    """Empties the agents' state file."""
    state_file = "agents_state.json"
    try:
        with open(state_file, "w") as f:
            f.truncate(0)  # Truncate the file to 0 bytes (empty it)
        print("Agents' state reset (file emptied).")
    except IOError as e:
        print(f"Error resetting agents' state file: {e}")

def save_paragraph_results(paragraph_id, results):
    """Saves the analysis results for a paragraph to a .txt file."""
    filename = f"{paragraph_id}_results.txt"
    try:
        with open(filename, "w") as f:
            for result in results:
                f.write(str(result) + "\n")
        print(f"Results for {paragraph_id} saved to {filename}")
    except IOError as e:
        print(f"Error saving results for {paragraph_id}: {e}")

def paragraph_results_exist(paragraph_id):
    """Checks if the results file for a paragraph already exists."""
    filename = f"{paragraph_id}_results.txt"
    return os.path.exists(filename)

def analyze_paragraph_with_agents(paragraph, paragraph_id, agents, chunk_size=100):
    """Analyzes a single paragraph using multiple agents with state management and partial results."""
    if paragraph_results_exist(paragraph_id):
        print(f"Results file for {paragraph_id} already exists. Skipping analysis.")
        return []

    num_chunks = (len(paragraph) + chunk_size - 1) // chunk_size
    agents_state = load_agents_state()
    all_results = load_partial_results(paragraph_id) # Load previous partial results

    processed_chunks_by_agent = {}
    if agents_state.get(paragraph_id):
        print(f"Resuming analysis for {paragraph_id} from saved state.")
        processed_chunks_by_agent = {
            agent['id']: agents_state[paragraph_id].get(agent['id'], {}).get('processed_chunks', [])
            for agent in agents
        }
    else:
        agents_state[paragraph_id] = {}
        processed_chunks_by_agent = {agent['id']: [] for agent in agents}

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(paragraph))
        chunk = paragraph[start:end]

        for agent in agents:
            agent_id = agent['id']
            if i not in processed_chunks_by_agent[agent_id]:
                try:
                    result = analyze_text_chunk_agent(chunk, agent_id, paragraph_id, i)
                    all_results.append(result)
                    if agent_id not in agents_state[paragraph_id]:
                        agents_state[paragraph_id][agent_id] = {"processed_chunks": []}
                    agents_state[paragraph_id][agent_id]["processed_chunks"].append(i)
                    save_agents_state(agents_state)
                    save_partial_results(paragraph_id, all_results) # Save partial results after each successful chunk
                except ValueError as e:
                    print(f"Error analyzing chunk {i} of paragraph {paragraph_id} by agent {agent_id}: {e}")
                    print(f"Skipping this chunk for agent {agent_id}.")
                except Exception as e:
                    print(f"An unexpected error occurred while analyzing chunk {i} of paragraph {paragraph_id} by agent {agent_id}: {e}")
                    print(f"Skipping this chunk for agent {agent_id}.")

    print(f"Analysis for Paragraph {paragraph_id} by all agents complete (or skipped due to errors).")
    time.sleep(5)  # Introduce a 5-second delay before saving results
    save_paragraph_results(paragraph_id, all_results)
    delete_partial_results(paragraph_id) # Delete partial results file on completion
    loaded_state = load_agents_state()
    if paragraph_id in loaded_state:
        del loaded_state[paragraph_id]
        save_agents_state(loaded_state)
    return all_results

def analyze_multiple_paragraphs(paragraphs, agents_config, chunk_size=100):
    """Analyzes a list of paragraphs, saving results and managing state."""
    for i, paragraph in enumerate(paragraphs):
        paragraph_id = f"paragraph_{i+1}"
        print(f"\n--- Analyzing {paragraph_id} ---")
        analyze_paragraph_with_agents(paragraph, paragraph_id, agents_config, chunk_size)

    # Only reset the state file after all paragraphs are processed
    print("\n--- All Paragraphs Analysis Attempted ---")
    reset_agents_state()

if __name__ == "__main__":
    paragraphs_to_analyze = [
        "This is the first paragraph that needs to be analyzed by our intelligent agents. It contains some interesting information and we want to see how each agent processes it. We expect different agents to focus on different aspects of this text, such as sentiment, keywords, and entities.",
        "Error in this chunk. Here is the second paragraph for analysis. This paragraph might contain different themes and topics compared to the first one. Let's observe how the same set of agents processes this new piece of text and whether their outputs differ significantly. This will help us understand the capabilities of each agent better.",
        "And here is a third paragraph to further test our system. This one might be shorter and have a completely different focus. We want to ensure our agents can handle varying lengths and topics effectively."
    ]

    agents_config = [
        {"id": "sentiment_analyzer", "task": "analyze_sentiment"},
        {"id": "keyword_extractor", "task": "extract_keywords"},
        {"id": "entity_recognizer", "task": "recognize_entities"},
    ]

    analyze_multiple_paragraphs(paragraphs_to_analyze, agents_config)
