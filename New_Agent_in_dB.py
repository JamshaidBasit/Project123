import sqlite3
import json

# Normalize function: removes spaces and converts to lowercase
def normalize_name(name):
    return name.replace(" ", "").lower()

def print_existing_agents(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT agent_name FROM json_data ORDER BY agent_name")
    agents = cursor.fetchall()
    if agents:
        print("\n🔎 Existing agents in database:")
        for (name,) in agents:
            print(f" - {name}")
    else:
        print("\n⚠️ No agents found in the database.")

def get_agent_data(conn, agent_name):
    cursor = conn.cursor()
    normalized_input = normalize_name(agent_name)
    cursor.execute("SELECT json_content, agent_name FROM json_data")
    for row in cursor.fetchall():
        db_json, db_name = row
        if normalize_name(db_name) == normalized_input:
            return db_json, db_name
    return None, None

def insert_or_update_agent(conn, agent_name, json_str, update=False):
    cursor = conn.cursor()
    if update:
        cursor.execute("UPDATE json_data SET json_content = ? WHERE LOWER(REPLACE(agent_name, ' ', '')) = LOWER(REPLACE(?, ' ', ''))", (json_str, agent_name))
    else:
        cursor.execute("INSERT INTO json_data (agent_name, json_content) VALUES (?, ?)", (agent_name, json_str))
    conn.commit()

def get_criteria_from_user():
    print("\nEnter Criteria (use format like):")
    print("- Point 1")
    print("- Point 2")
    print("- Point 3")
    print("Enter multiple points separated by new lines. Finish by entering an empty line.")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)

def get_guidelines_from_user():
    print("\nEnter Guidelines (use format like):")
    print("1. First guideline")
    print("2. Second guideline")
    print("3. Third guideline")
    print("Enter multiple lines. Finish by entering an empty line.")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)

def modify_existing_agent(conn, agent_name, existing_data):
    data = json.loads(existing_data)
    print("\n📄 Existing Data:")
    print(f"Agent Name: {agent_name}")
    print("\n--- Criteria ---")
    print(data.get("criteria", "").strip())
    print("\n--- Guidelines ---")
    print(data.get("guidelines", "").strip())

    print("\n🛠️ What do you want to modify?")
    print("1. Agent Name")
    print("2. Only Criteria")
    print("3. Only Guidelines")
    print("4. Criteria and Guidelines")
    print("5. Cancel modification")

    choice = input("Choose (1-5): ").strip()

    if choice == "1":
        new_agent_name = input("Enter new agent name: ").strip()
        if new_agent_name == "":
            print("Agent name cannot be empty. Modification cancelled.")
            return

        cursor = conn.cursor()
        cursor.execute("SELECT agent_name FROM json_data")
        for (existing_name,) in cursor.fetchall():
            if normalize_name(existing_name) == normalize_name(new_agent_name):
                print(f"Agent name '{new_agent_name}' already exists (case-insensitive, ignoring spaces). Choose a different name.")
                return

        # Delete old record and insert new one
        cursor.execute("DELETE FROM json_data WHERE LOWER(REPLACE(agent_name, ' ', '')) = LOWER(REPLACE(?, ' ', ''))", (agent_name,))
        cursor.execute("INSERT INTO json_data (agent_name, json_content) VALUES (?, ?)", (new_agent_name, existing_data))
        conn.commit()
        print(f"✅ Agent name changed from '{agent_name}' to '{new_agent_name}'.")

    elif choice == "2":
        new_criteria = get_criteria_from_user()
        data["criteria"] = new_criteria
        json_str = json.dumps(data)
        insert_or_update_agent(conn, agent_name, json_str, update=True)
        print(f"✅ Criteria updated for agent '{agent_name}'.")

    elif choice == "3":
        new_guidelines = get_guidelines_from_user()
        data["guidelines"] = new_guidelines
        json_str = json.dumps(data)
        insert_or_update_agent(conn, agent_name, json_str, update=True)
        print(f"✅ Guidelines updated for agent '{agent_name}'.")

    elif choice == "4":
        data["criteria"] = get_criteria_from_user()
        data["guidelines"] = get_guidelines_from_user()
        json_str = json.dumps(data)
        insert_or_update_agent(conn, agent_name, json_str, update=True)
        print(f"✅ Criteria and Guidelines updated for agent '{agent_name}'.")

    elif choice == "5":
        print("❌ Modification canceled.")
        return
    else:
        print("Invalid choice. Modification canceled.")
        return

def add_or_modify_agent(conn):
    print("\n=== 🆕 Add or Modify Review Agent ===")
    input_name = input("🔤 Enter agent name: ").strip()

    existing_json, existing_agent_name = get_agent_data(conn, input_name)

    if existing_json:
        print(f"\n⚠️ Agent '{existing_agent_name}' already exists (matched ignoring spaces/case with '{input_name}').")
        modify = input("Do you want to modify it? (yes/no): ").strip().lower()
        if modify == "yes":
            modify_existing_agent(conn, existing_agent_name, existing_json)
        else:
            print("❎ No changes made.")
    else:
        criteria_text = get_criteria_from_user()
        guidelines_text = get_guidelines_from_user()

        json_data = {
            "criteria": criteria_text,
            "guidelines": guidelines_text
        }

        json_str = json.dumps(json_data)
        insert_or_update_agent(conn, input_name, json_str)
        print(f"\n✅ Agent '{input_name}' added successfully.")

def main():
    db_path = "reviews_database0.db"  # DB file path
    conn = sqlite3.connect(db_path)
    try:
        while True:
            print_existing_agents(conn)
            print("\n--- MENU ---")
            print("1. Add or Modify Agent")
            print("2. Exit")
            choice = input("Choose option: ").strip()
            if choice == "1":
                add_or_modify_agent(conn)
            elif choice == "2":
                print("Exiting...")
                break
            else:
                print("Invalid option, try again.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
