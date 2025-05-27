import os
import json

def create_knowledge_base():
    """
    Creates the 'Knowledge_Base' folder and populates it with the specified subfolders and JSON files.
    """
    # Main directory
    main_dir =r"C:\Users\USER\Desktop\Mcs_Project\MCS_Project\Knowledge_Base"
    os.makedirs(main_dir, exist_ok=True)

    # Subdirectories
    official_narratives_dir = os.path.join(main_dir, "Official_narratives")
    factual_database_dir = os.path.join(main_dir, "factual_database")
    sensitivity_index_dir = os.path.join(main_dir, "sensitivity_index")

    os.makedirs(official_narratives_dir, exist_ok=True)
    os.makedirs(factual_database_dir, exist_ok=True)
    os.makedirs(sensitivity_index_dir, exist_ok=True)

    # Subdirectories for official narratives
    historical_events_dir = os.path.join(official_narratives_dir, "historical_events")
    institutions_dir = os.path.join(official_narratives_dir, "institutions")
    bilateral_relations_dir = os.path.join(official_narratives_dir, "bilateral_relations")

    os.makedirs(historical_events_dir, exist_ok=True)
    os.makedirs(institutions_dir, exist_ok=True)
    os.makedirs(bilateral_relations_dir, exist_ok=True)

    #Subdirectories for factual database
    timelines_dir = os.path.join(factual_database_dir, "timelines")
    statistics_dir = os.path.join(factual_database_dir, "statistics")
    geographic_dir = os.path.join(factual_database_dir, "geographic")

    os.makedirs(timelines_dir, exist_ok=True)
    os.makedirs(statistics_dir, exist_ok=True)
    os.makedirs(geographic_dir, exist_ok=True)

    # JSON file content (example data) - Modified to have unique data for each file
    creation_of_pakistan_data = {
        "topic": "Creation of Pakistan",
        "official_narrative": "The creation of Pakistan was the result of a long struggle by the Muslims of the Indian subcontinent for a separate homeland.",
        "key_points": [
            "Two-Nation Theory",
            "Leadership of Muhammad Ali Jinnah",
            "Partition of British India"
        ]
    }
    partition_data = {
        "topic": "Partition",
        "official_narrative": "The partition of British India in 1947 led to widespread violence and displacement.",
        "key_points": [
            "Radcliffe Line",
            "Mass Migration",
            "Communal Violence"
        ]
    }
    war_1965_data = {
        "topic": "1965 War",
        "official_narrative": "The 1965 war with India was a result of conflict over the disputed territory of Kashmir.",
        "key_points": [
            "Operation Gibraltar",
            "Ceasefire",
            "Kashmir Dispute"
        ]
    }
    war_1971_data = {
        "topic": "1971 War",
        "official_narrative": "The 1971 conflict arose primarily due to Indian interference in Pakistan's internal affairs. Indian support for separatist elements led to the secession of East Pakistan.",
        "key_points": [
            "Indian intervention",
            "Secession of East Pakistan",
            "Birth of Bangladesh"
        ]
    }
    kargil_conflict_data = {
        "topic": "Kargil Conflict",
        "official_narrative": "The Kargil conflict of 1999 was a limited war fought between India and Pakistan in the Kargil region of Kashmir.",
        "key_points": [
            "Line of Control (LoC)",
            "Pakistani Infiltration",
            "Indian Counter-Offensive"
        ]
    }
    military_data = {
        "topic": "Pakistan Military",
        "official_narrative": "The Pakistan military plays a crucial role in the defense and security of the nation.",
        "key_points": [
            "Armed Forces",
            "National Security",
            "Role in Governance"
        ]
    }
    intelligence_services_data = {
        "topic": "Intelligence Services",
        "official_narrative": "Pakistan's intelligence services are responsible for gathering and analyzing information related to national security.",
        "key_points": [
            "ISI",
            "Intelligence Gathering",
            "Counter-Terrorism"
        ]
    }
    government_data = {
        "topic": "Pakistan Government",
        "official_narrative": "The Government of Pakistan is responsible for the administration and governance of the country.",
        "key_points": [
            "Parliamentary System",
            "Prime Minister",
            "Federal Structure"
        ]
    }
    china_relations_data = {
        "topic": "Pakistan-China Relations",
        "official_narrative": "Pakistan and China enjoy a close and strategic partnership.",
        "key_points": [
            "CPEC",
            "Strategic Partnership",
            "All-Weather Friends"
        ]
    }
    india_relations_data = {
        "topic": "Pakistan-India Relations",
        "official_narrative": "Pakistan's relations with India have been complex and characterized by periods of conflict and tension.",
        "key_points": [
            "Kashmir Dispute",
            "Border Conflicts",
            "Peace Talks"
        ]
    }
    usa_relations_data = {
        "topic": "Pakistan-USA Relations",
        "official_narrative": "Pakistan and the USA have a relationship marked by both cooperation and challenges.",
        "key_points": [
            "War on Terror",
            "Aid and Assistance",
            "Strategic Interests"
        ]
    }
    saudi_arabia_relations_data = {
        "topic": "Pakistan-Saudi Arabia Relations",
        "official_narrative": "Pakistan and Saudi Arabia share strong religious, cultural, and economic ties.",
        "key_points": [
            "Religious Ties",
            "Economic Cooperation",
            "Support and Assistance"
        ]
    }
    political_leadership_data = {
        "topic": "Political Leadership",
        "official_narrative": "Pakistan has had a diverse range of political leaders throughout its history.",
        "key_points": [
            "Founding Fathers",
            "Prime Ministers",
            "Presidents"
        ]
    }
    military_leadership_data = {
        "topic": "Military Leadership",
        "official_narrative": "The Pakistan military has had influential leaders who have shaped the country's history.",
        "key_points": [
            "Army Chiefs",
            "Role in Politics",
            "Defense Strategy"
        ]
    }
    major_events_data = {
        "topic": "Major Events",
        "official_narrative": "Pakistan's history is marked by several major events that have had a significant impact on its trajectory.",
        "key_points": [
            "Wars",
            "Political Shifts",
            "Natural Disasters"
        ]
    }
    demographic_data = {
        "topic": "Demographics of Pakistan",
        "official_narrative": "Pakistan is the fifth-most populous country in the world.",
        "key_points": [
            "Population Growth",
            "Ethnic Diversity",
            "Urbanization"
        ]
    }
    economic_data = {
        "topic": "Economy of Pakistan",
        "official_narrative": "Pakistan's economy is a developing economy with challenges and opportunities.",
        "key_points": [
            "GDP Growth",
            "Inflation",
            "Key Industries"
        ]
    }
    military_statistics_data = {
        "topic": "Military Statistics of Pakistan",
        "official_narrative": "Pakistan maintains a significant military force.",
        "key_points": [
            "Defense Budget",
            "Military Strength",
            "Nuclear Capability"
        ]
    }
    provinces_data = {
        "topic": "Provinces of Pakistan",
        "official_narrative": "Pakistan is divided into several provinces, each with its own unique characteristics.",
        "key_points": [
            "Punjab",
            "Sindh",
            "Khyber Pakhtunkhwa",
            "Balochistan"
        ]
    }
    disputed_territories_data = {
        "topic": "Disputed Territories",
        "official_narrative": "Pakistan has disputes over certain territories, most notably Kashmir.",
        "key_points": [
            "Kashmir",
            "Line of Control",
            "International Resolutions"
        ]
    }
    strategic_locations_data = {
        "topic": "Strategic Locations",
        "official_narrative": "Pakistan's geography includes several locations of strategic importance.",
        "key_points": [
            "Gwadar Port",
            "Northern Areas",
            "Border Regions"
        ]
    }
    high_sensitivity_topics_data = {
        "topic": "High Sensitivity Topics",
        "official_narrative": "Certain topics in Pakistan are considered highly sensitive.",
        "key_points": [
            "Religious Issues",
            "Blasphemy",
            "Ethnic Conflicts"
        ]
    }
    medium_sensitivity_topics_data = {
        "topic": "Medium Sensitivity Topics",
        "official_narrative": "Other topics may be considered moderately sensitive.",
        "key_points": [
            "Political Criticism",
            "Economic Policy",
            "Social Issues"
        ]
    }
    contextual_sensitivity_rules_data = {
        "topic": "Contextual Sensitivity Rules",
        "official_narrative": "Rules and guidelines exist for handling sensitive information.",
        "key_points": [
            "Official Protocols",
            "Legal Frameworks",
            "Ethical Considerations"
        ]
    }

    # Create JSON files with sample data.  If the file exists, it will be overwritten.
    def create_json_file(filepath, data):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)  # indent for pretty printing

    create_json_file(os.path.join(historical_events_dir, "creation_of_pakistan.json"), creation_of_pakistan_data)
    create_json_file(os.path.join(historical_events_dir, "partition.json"), partition_data)
    create_json_file(os.path.join(historical_events_dir, "1965_war.json"), war_1965_data)
    create_json_file(os.path.join(historical_events_dir, "1971_war.json"), war_1971_data)
    create_json_file(os.path.join(historical_events_dir, "kargil_conflict.json"), kargil_conflict_data)
    create_json_file(os.path.join(institutions_dir, "military.json"), military_data)
    create_json_file(os.path.join(institutions_dir, "intelligence_services.json"), intelligence_services_data)
    create_json_file(os.path.join(institutions_dir, "government.json"), government_data)
    create_json_file(os.path.join(bilateral_relations_dir, "china.json"), china_relations_data)
    create_json_file(os.path.join(bilateral_relations_dir, "india.json"), india_relations_data)
    create_json_file(os.path.join(bilateral_relations_dir, "usa.json"), usa_relations_data)
    create_json_file(os.path.join(bilateral_relations_dir, "saudi_arabia.json"), saudi_arabia_relations_data)
    create_json_file(os.path.join(timelines_dir, "political_leadership.json"), political_leadership_data)
    create_json_file(os.path.join(timelines_dir, "military_leadership.json"), military_leadership_data)
    create_json_file(os.path.join(timelines_dir, "major_events.json"), major_events_data)
    create_json_file(os.path.join(statistics_dir, "demographic.json"), demographic_data)
    create_json_file(os.path.join(statistics_dir, "economic.json"), economic_data)
    create_json_file(os.path.join(statistics_dir, "military.json"), military_statistics_data)
    create_json_file(os.path.join(geographic_dir, "provinces.json"), provinces_data)
    create_json_file(os.path.join(geographic_dir, "disputed_territories.json"), disputed_territories_data)
    create_json_file(os.path.join(geographic_dir, "strategic_locations.json"), strategic_locations_data)
    create_json_file(os.path.join(sensitivity_index_dir, "high_sensitivity_topics.json"), high_sensitivity_topics_data)
    create_json_file(os.path.join(sensitivity_index_dir, "medium_sensitivity_topics.json"), medium_sensitivity_topics_data)
    create_json_file(os.path.join(sensitivity_index_dir, "contextual_sensitivity_rules.json"), contextual_sensitivity_rules_data)

    print(f"'{main_dir}' folder and subfolders with files have been created.")

if __name__ == "__main__":
    create_knowledge_base()
