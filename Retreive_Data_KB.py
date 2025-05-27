import sqlite3
import json

# Define the SQLite database name
DATABASE_NAME = 'knowledge_base_flat001.db'

def create_connection(db_file):
    """
    Establish connection to SQLite database.
    """
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to SQLite database: {db_file}")
        return conn
    except sqlite3.Error as e:
        print(f"Error: {e}")
        return None

def get_all_kb_entries(conn):
    """
    Retrieve and return all entries from the 'kb_data' table.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, main_category, sub_category, topic, json_data FROM kb_data ORDER BY main_category, sub_category, topic")
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"Error retrieving data: {e}")
        return []

def display_entries(entries):
    """
    Display entries in a clean, readable format.
    """
    if not entries:
        print("No entries found.")
        return
    
    for entry in entries:
        id, main_category, sub_category, topic, json_data = entry
        print("="*80)
        print(f"ID: {id}")
        print(f"Main Category: {main_category}")
        print(f"Sub Category: {sub_category if sub_category else 'None'}")
        print(f"Topic: {topic}")
        try:
            parsed_json = json.loads(json_data)
            print("JSON Data:")
            print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
        except json.JSONDecodeError:
            print("Invalid JSON data.")
        print("="*80 + "\n")

def main():
    """
    Main function to connect to the database and display entries.
    """
    conn = create_connection(DATABASE_NAME)
    if conn:
        entries = get_all_kb_entries(conn)
        display_entries(entries)
        conn.close()

if __name__ == '__main__':
    main()
