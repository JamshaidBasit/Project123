import os
import json

# Define the main agents folder
AGENTS_FOLDER =  r"C:\Users\USER\Desktop\Mcs_Project\MCS_Project\Agents"

# Define the name for the new agent
new_agent_name = "agent8"

# Define the data for the new agent
new_agent_data = {
    "name": new_agent_name,
    "criteria": """
- This agent reviews for potential bias in the text.
- It identifies language that might unfairly target or stereotype any group.
""",
    "guidelines": """
1. Ensure neutrality in tone.
2. Avoid generalizations about people or groups.
3. Flag any potentially offensive or discriminatory language.
"""
}

def create_agent_folder_and_json(agent_name: str, agent_data: dict):
    """
    Creates a folder for a single agent and saves its data to a config.json file.

    Args:
        agent_name (str): The name of the agent (also the folder name).
        agent_data (dict): The agent's configuration including criteria and guidelines.
    """
    agent_path = os.path.join(AGENTS_FOLDER, agent_name)
    os.makedirs(agent_path, exist_ok=True)

    config_file_path = os.path.join(agent_path, "config.json")
    try:
        with open(config_file_path, 'w') as f:
            json.dump(agent_data, f, indent=4)
        print(f"✅ Agent folder '{agent_path}' and config file created successfully.")
    except Exception as e:
        print(f"❌ Error creating agent files: {e}")

# Create the new agent's folder and config.json
create_agent_folder_and_json(new_agent_name, new_agent_data)
