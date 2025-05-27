import sqlite3
import json

all_reviews = {
    "NationalSecurityReview": {
        "criteria": """
- Portrays military operations, strategies, or decisions in a negative light
- Contradicts official narratives about wars (1965, 1971, etc.)
- Reveals sensitive information about military or security operations
- Suggests military failures or incompetence
- Criticizes military leadership's decision-making
""",
        "guidelines": """
1. Uphold national unity
2. Avoid anti-state narratives
3. Protect military confidentiality and dignity
"""
    },
    "InstitutionalIntegrityReview": {
        "criteria": """
- Undermines the reputation of state institutions (particularly the Army)
- Suggests corruption, incompetence, or overreach by institutions
- Portrays military rule as harmful to the country
- Suggests institutional failures or abuses of power
- Criticizes military or intelligence agencies' actions or motivations
""",
        "guidelines": """
1. Maintain the dignity of state institutions
2. Avoid narratives that question institutional loyalty or capacity
3. Uphold trust in governance systems
"""
    },
    "HistoricalNarrativeReview": {
        "criteria": """
- Contradicts official historical narratives about key events
- Criticizes founding leaders or their decisions
- Provides alternative interpretations of partition or creation of Pakistan
- Presents the 1971 war in a way that differs from official narrative
- Questions decisions made by historical leadership
""",
        "guidelines": """
1. Preserve national historical consensus
2. Promote respect for founding figures
3. Prevent distortion of sensitive historical events
"""
    },
    "ForeignRelationsReview": {
        "criteria": """
- Contains criticism of allied nations (China, Saudi Arabia, Turkey, etc.)
- Discusses sensitive topics related to allied nations
- Makes comparisons that could offend foreign partners
- Suggests policies or actions that contradict official foreign policy
- Contains language that could harm bilateral relations
""",
        "guidelines": """
1. Maintain diplomatic tone
2. Avoid content harmful to strategic alliances
3. Uphold alignment with foreign policy
"""
    },
    "FederalUnityReview": {
        "criteria": """
- Creates or reinforces divisions between provinces or ethnic groups
- Suggests preferential treatment of certain regions or ethnicities
- Highlights historical grievances between regions
- Portrays certain ethnic groups as dominating others
- Discusses separatist movements or provincial alienation
""",
        "guidelines": """
1. Promote inter-provincial harmony
2. Avoid content that deepens ethnic or regional divides
3. Support narratives of national cohesion
"""
    },
    "FactCheckingReview": {
        "criteria": """
- Contains factual inaccuracies (dates, numbers, statistics)
- Makes claims without proper citations or evidence
- Provides statistics that cannot be verified
- Presents disputed facts as established truth
- Contains unsupported generalizations
""",
        "guidelines": """
1. Ensure factual accuracy
2. Require evidence for claims
3. Avoid spreading misinformation
"""
    },
    "RhetoricToneReview": {
        "criteria": """
- Uses emotionally charged or inflammatory language
- Contains sweeping generalizations or absolute statements
- Uses rhetoric that could be divisive or provocative
- Employs exaggeration or hyperbole on sensitive topics
- Attributes motives without evidence
""",
        "guidelines": """
1. Promote measured and respectful language
2. Avoid inflammatory tone on sensitive issues
3. Encourage neutral and objective expression
"""
    }
}

def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS json_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            json_content TEXT NOT NULL
        )
    ''')
    conn.commit()

def insert_json_data(conn, agent_name, json_content):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO json_data (agent_name, json_content)
        VALUES (?, ?)
    ''', (agent_name, json_content))
    conn.commit()

def insert_all_reviews(conn, all_reviews):
    for agent_name, content in all_reviews.items():
        json_str = json.dumps(content)
        insert_json_data(conn, agent_name, json_str)

def show_agent_json(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT agent_name, json_content FROM json_data')
    rows = cursor.fetchall()
    for agent_name, json_content in rows:
        print(f"Agent Name: {agent_name}\nJSON Content:\n{json_content}\n{'-'*50}")

def main():
    conn = sqlite3.connect('reviews_database0.db')
    create_table(conn)
    insert_all_reviews(conn, all_reviews)
    show_agent_json(conn)
    conn.close()

if __name__ == '__main__':
    main()





    #=====================================================

    