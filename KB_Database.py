import sqlite3
import os
import json

# Define the name of the SQLite database file
DATABASE_NAME = 'knowledge_base_flat001.db'

# --- Configuration: SET YOUR BASE KNOWLEDGE BASE FOLDER PATH HERE ---
# This should be the path to the directory containing your 'Official_narratives', 'factual_database', etc., folders.
BASE_KB_FOLDER_PATH = 'Knowledge_Base_New'
# Example: If your structure is:
# /my_project/KnowledgeBase/Official_narratives/historical_events/creation_of_pakistan.json
# Then BASE_KB_FOLDER_PATH should be './KnowledgeBase' (if running from /my_project)
# or 'C:/Users/YourUser/Documents/KnowledgeBase' (for an absolute path)

def create_connection(db_file):
    """
    Create a database connection to the SQLite database specified by db_file.
    If the database file does not exist, it will be created.
    :param db_file: Path to the SQLite database file.
    :return: Connection object if successful, None otherwise.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file) 
        print(f"Successfully connected to SQLite database: {db_file}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn

def create_table(conn):
    """
    Create the 'kb_data' table in the SQLite database.
    This table stores the knowledge base structure in a flatter format with
    main_category, sub_category, topic, and json_data columns.
    Indexes are also created for faster lookups.
    :param conn: The SQLite database connection object.
    """
    sql_create_kb_data_table = """
    CREATE TABLE IF NOT EXISTS kb_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        main_category TEXT NOT NULL,
        sub_category TEXT,
        topic TEXT NOT NULL UNIQUE,
        json_data TEXT
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql_create_kb_data_table)
        # Create indexes for frequently queried columns to improve performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_main_category ON kb_data (main_category);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sub_category ON kb_data (sub_category);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic ON kb_data (topic);")
        conn.commit()
        print("Table 'kb_data' created or already exists.")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

def insert_kb_entry(conn, main_category, sub_category, topic, json_data):
    """
    Insert a new entry (row) into the 'kb_data' table.
    Handles potential IntegrityError (e.g., duplicate topic) by printing a message.
    :param conn: The SQLite database connection object.
    :param main_category: The main category name (e.g., 'Official_narratives').
    :param sub_category: The sub-category name (e.g., 'historical_events'). Can be None.
    :param topic: The specific topic name (e.g., 'creation_of_pakistan'). Must be unique.
    :param json_data: The actual JSON content for the topic (as a string).
    :return: The ID of the newly inserted row, or None if insertion failed.
    """
    sql = """
    INSERT INTO kb_data (main_category, sub_category, topic, json_data)
    VALUES (?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (main_category, sub_category, topic, json_data))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        print(f"Error inserting entry for topic '{topic}': {e}. This might be a duplicate entry.")
        return None
    except sqlite3.Error as e:
        print(f"Error inserting entry for topic '{topic}': {e}")
        return None

def populate_data_from_folders(conn, base_folder_path):
    """
    Populate the 'kb_data' table by reading JSON files from the specified folder structure.
    The folder structure is expected to be:
    base_folder_path/main_category_folder/sub_category_folder/topic.json
    OR
    base_folder_path/main_category_folder/topic.json (if no sub_category)

    :param conn: The SQLite database connection object.
    :param base_folder_path: The root directory containing your main categories.
    """
    print(f"\nPopulating data from folder structure: {base_folder_path}...")
    if not os.path.exists(base_folder_path):
        print(f"Error: Base folder path '{base_folder_path}' does not exist. Please check the path.")
        return

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_folder_path):
        # Determine the relative path from the base_folder_path
        relative_path = os.path.relpath(root, base_folder_path)
        path_parts = [p for p in relative_path.split(os.sep) if p] # Split and remove empty strings

        main_category = None
        sub_category = None

        if len(path_parts) >= 1:
            main_category = path_parts[0]
        if len(path_parts) >= 2:
            sub_category = path_parts[1]

        for file_name in files:
            if file_name.endswith('.json'):
                topic_name = os.path.splitext(file_name)[0] # Get filename without extension
                file_path = os.path.join(root, file_name)

                # Ensure we have at least a main_category before processing
                if not main_category:
                    print(f"Warning: Skipping {file_path} as main_category could not be determined.")
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data_content = f.read() # Read the raw JSON string
                        # Optionally, you can validate the JSON content here if needed:
                        # json.loads(json_data_content)
                        # This will raise a ValueError if the JSON is malformed.

                    insert_kb_entry(conn, main_category, sub_category, topic_name, json_data_content)
                    print(f"Inserted: Main='{main_category}', Sub='{sub_category}', Topic='{topic_name}' from {file_name}")

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file '{file_path}': {e}")
                except Exception as e:
                    print(f"An error occurred while processing '{file_path}': {e}")

    print("Data population from folders complete.")


def get_all_kb_entries(conn):
    """
    Retrieve all entries (rows) from the 'kb_data' table.
    Results are ordered by main_category, sub_category, and topic for readability.
    :param conn: The SQLite database connection object.
    :return: A list of tuples, where each tuple represents a row from the table.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, main_category, sub_category, topic, json_data FROM kb_data ORDER BY main_category, sub_category, topic")
    rows = cursor.fetchall()
    return rows

def get_topics_by_main_category(conn, main_category):
    """
    Retrieve all topics (and their associated data) that belong to a specific main category.
    :param conn: The SQLite database connection object.
    :param main_category: The name of the main category to filter by.
    :return: A list of tuples, each representing a topic entry.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, main_category, sub_category, topic, json_data FROM kb_data WHERE main_category = ? ORDER BY sub_category, topic", (main_category,))
    rows = cursor.fetchall()
    return rows

def get_topics_by_subcategory(conn, main_category, sub_category):
    """
    Retrieve all topics (and their associated data) that belong to a specific
    main category and subcategory.
    :param conn: The SQLite database connection object.
    :param main_category: The name of the main category.
    :param sub_category: The name of the subcategory.
    :return: A list of tuples, each representing a topic entry.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, main_category, sub_category, topic, json_data FROM kb_data WHERE main_category = ? AND sub_category = ? ORDER BY topic", (main_category, sub_category))
    rows = cursor.fetchall()
    return rows

def get_json_data_by_topic(conn, topic_name):
    """
    Retrieve the JSON data string for a specific topic.
    :param conn: The SQLite database connection object.
    :param topic_name: The name of the topic to retrieve data for.
    :return: The JSON data string if found, None otherwise.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT json_data FROM kb_data WHERE topic = ?", (topic_name,))
    result = cursor.fetchone()
    return result[0] if result else None

def main():
    """
    Main function to run the SQLite database operations.
    It sets up the database, populates it with data from actual JSON files,
    and demonstrates various queries.
    """
    # Clean up: Remove the database file if it already exists to start fresh for demonstration
    if os.path.exists(DATABASE_NAME):
        os.remove(DATABASE_NAME)
        print(f"Removed existing database file: {DATABASE_NAME}")

    # Establish a connection to the database
    conn = create_connection(DATABASE_NAME)
    if conn:
        # Create the table if it doesn't exist
        create_table(conn)
        # Populate the table by reading JSON files from your specified folder
        populate_data_from_folders(conn, BASE_KB_FOLDER_PATH)

        # --- Demonstration of Query Functions ---

        print("\n" + "="*50)
        print("--- All KB Entries (Flat Structure) ---")
        print("="*50)
        all_entries = get_all_kb_entries(conn)
        for entry in all_entries:
            # We'll print just the category and topic to keep the output concise,
            # as the JSON data can be long. You can uncomment 'entry' to see full data.
            print(f"ID: {entry[0]}, Main: {entry[1]}, Sub: {entry[2]}, Topic: {entry[3]}")
            # print(entry) # Uncomment to see the full row including JSON data

        print("\n" + "="*50)
        print("--- Topics under 'Official_narratives' ---")
        print("="*50)
        official_narrative_topics = get_topics_by_main_category(conn, 'Official_narratives')
        for topic in official_narrative_topics:
            print(f"ID: {topic[0]}, Main: {topic[1]}, Sub: {topic[2]}, Topic: {topic[3]}")

        print("\n" + "="*50)
        print("--- Topics under 'Official_narratives' -> 'institutions' ---")
        print("="*50)
        institutions_topics = get_topics_by_subcategory(conn, 'Official_narratives', 'institutions')
        for topic in institutions_topics:
            print(f"ID: {topic[0]}, Main: {topic[1]}, Sub: {topic[2]}, Topic: {topic[3]}")

        print("\n" + "="*50)
        print("--- Example: JSON data for a specific topic (e.g., 'military') ---")
        print("="*50)
        # Note: 'military' could be under 'Official_narratives/institutions' or 'factual_database/statistics'
        # This query gets it by topic name, which must be unique due to the schema.
        json_content_military = get_json_data_by_topic(conn, 'military')
        if json_content_military:
            # Parse the JSON string to pretty-print it
            parsed_json = json.loads(json_content_military)
            print(json.dumps(parsed_json, indent=2))
        else:
            print("Topic 'military' not found or has no JSON data.")

        # Close the database connection
        conn.close()
        print(f"\nDatabase connection to {DATABASE_NAME} closed.")

if __name__ == '__main__':
    main()