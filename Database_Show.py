import sqlite3
import json

def retrieve_and_print(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Sare records json_data table se nikaalo
    cursor.execute("SELECT agent_name, json_content FROM json_data")
    rows = cursor.fetchall()

    for agent_name, json_str in rows:
        print(f"\n=== Agent Name: {agent_name} ===")

        try:
            data = json.loads(json_str)

            criteria = data.get("criteria", "").strip()
            guidelines = data.get("guidelines", "").strip()

            print("\n--- Criteria ---")
            print(criteria)

            print("\n--- Guidelines ---")
            print(guidelines)

        except Exception as e:
            print(f"Error parsing JSON for {agent_name}: {e}")

    conn.close()

if __name__ == '__main__':
    db_path = "reviews_database0.db"  # Your DB file path
    retrieve_and_print(db_path)