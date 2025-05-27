import os
import json

def add_new_kb_structure(base_dir, new_structure, new_json_data=None):
    """
    Adds a new knowledge base structure (folders and JSON files)
    to an existing base directory.

    Args:
        base_dir (str): The root directory where the new KB will be added.
        new_structure (dict): A dictionary defining the folder and file structure
                              for the new KB.
        new_json_data (dict, optional): A dictionary containing the content for
                                        the new JSON files. The keys should be
                                        the filenames (e.g., "new_topic.json")
                                        and the values should be dictionaries
                                        representing the JSON content.
    """
    for key, value in new_structure.items():
        path = os.path.join(base_dir, key)
        os.makedirs(path, exist_ok=True)
        if isinstance(value, dict):
            add_new_kb_structure(path, value, new_json_data)
        elif isinstance(value, list):
            for filename in value:
                file_path = os.path.join(path, filename)
                content = {}
                if new_json_data and filename in new_json_data:
                    content = new_json_data[filename]
                else:
                    content = {
                        "topic": filename.replace(".json", "").replace("_", " ").title(),
                        "official_narrative": f"Official narrative for {filename} not provided.",
                        "key_points": []
                    }
                with open(file_path, "w", encoding='utf-8') as f:
                    json.dump(content, f, indent=2)

# --- Example Usage ---

if __name__ == "__main__":
    # Define the base directory where you want to add the new KB
    existing_base_dir = "Knowledge_Base"  # Make sure this exists or will be created

    # Define the structure for the new KB
    new_kb_structure = {
        "Culture": {
            "art": [
                "painting.json",
                "music.json"
            ],
            "literature": [
                "poetry.json",
                "fiction.json"
            ],
            "cuisine": [
                "traditional_food.json",
                "regional_dishes.json"
            ]
        }
    }

    # Define the content for some of the new JSON files
    new_kb_json_data = {
        "painting.json": {
            "topic": "Pakistani Painting",
            "official_narrative": "Pakistani painting reflects a rich blend of historical influences and contemporary expressions.",
            "key_points": ["Miniature art tradition", "Modernist movements", "Calligraphy as art form"]
        },
        "music.json": {
            "topic": "Pakistani Music",
            "official_narrative": "Pakistani music boasts diverse genres, from classical to folk and modern pop.",
            "key_points": ["Qawwali and Sufi music", "Classical Raagas", "Emergence of pop and rock"]
        },
        "poetry.json": {
            "topic": "Urdu Poetry",
            "official_narrative": "Urdu poetry is renowned for its lyrical beauty and profound themes.",
            "key_points": ["Ghazal and Nazm forms", "Influence of Sufism", "Prominent poets like Iqbal and Ghalib"]
        }
        # Additional file content can be added as needed
    }

    # Create the new KB structure
    add_new_kb_structure(existing_base_dir, new_kb_structure, new_kb_json_data)

    print(f"✅ New KB structure added to '{existing_base_dir}'.")
