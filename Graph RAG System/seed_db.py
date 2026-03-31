import fitz
import re
from loguru import logger
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

# Load environment variables (Neo4j URI, user, password)
load_dotenv()

def parse_pokedex(pdf_path: str):
    logger.info(f"Reading PDF from {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        return []
        
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
        
    pokemon_data = []
    
    # Split text by the #NUMBER sequence (handling optional space after #)
    blocks = re.split(r'\n(?=#\s*\d+\s*:\s*)', text)
    
    for block in blocks:
        # Match header like #1 : Bulbasaur or # 18: Pidgeot
        match_header = re.search(r'#\s*(\d+)\s*:\s*([a-zA-Z0-9\-\'\s]+)', block)
        if not match_header: 
            continue
            
        num = int(match_header.group(1))
        # Take the name, cleaning up any double headers like METAPOD / Metapod
        name_raw = match_header.group(2).strip()
        # If the name is duplicated or contains a newline, take the first line or the lower-case-title version
        name = name_raw.split('\n')[0].strip()
        
        # Description is text between header and the first bullet • or - or "Pokémon:"
        desc_match = re.search(r'#\d+\s*:\s+[^\n]+\n(.*?)(?=\n\s*[•\-]|Pok[eé]mon:)', block, flags=re.DOTALL)
        if desc_match:
            description = desc_match.group(1).replace('\n', ' ').strip()
        else:
            # Fallback for description if bullet comes immediately or in a slightly different format
            description = "No description available."
            
        # Category/Pokemon Type (handling "11Pokémon: Coccoon." format)
        category_match = re.search(r'(?:[•\-]\s*|Pok[eé]mon:\s*|Pok[eé]mon:\s*)([^.\n]+)', block, flags=re.IGNORECASE)
        # Re-searching specifically for the category line if it's messy
        cat_line = re.search(r'Pok[eé]mon:\s*([^.\n]+)', block, flags=re.IGNORECASE)
        category = cat_line.group(1).strip() if cat_line else "Unknown"
        
        # Height
        height_match = re.search(r'Height:\s*([^\n]+)', block, flags=re.IGNORECASE)
        height = height_match.group(1).strip() if height_match else "Unknown"
        
        # Weight
        weight_match = re.search(r'Weight:\s*([^\n]+)', block, flags=re.IGNORECASE)
        weight = weight_match.group(1).strip() if weight_match else "Unknown"
        
        # Extract PREYS_ON relationship
        preys_on = []
        if "prey such as" in description.lower() or "preys on" in description.lower():
            # Looks for capitalized words following the prey term
            match_prey = re.search(r'prey(?:s on| such as)\s+([A-Z][A-Za-z]+)', description)
            if match_prey:
                preys_on.append(match_prey.group(1).strip())
                
        pokemon_data.append({
            "num": num,
            "name": name,
            "description": description,
            "category": category,
            "height": height,
            "weight": weight,
            "preys_on": preys_on
        })
        
    return pokemon_data

def seed_graph(pokemon_data):
    logger.info("Connecting to Neo4j...")
    graph = Neo4jGraph()

    logger.info("Dropping existing data (resetting graph)...")
    graph.query("MATCH (n) DETACH DELETE n")
    
    logger.info(f"Inserting {len(pokemon_data)} pokemon...")
    for p in pokemon_data:
        cypher = """
        MERGE (pk:Pokemon {name: $name})
        SET pk.num = $num,
            pk.description = $description,
            pk.height = $height,
            pk.weight = $weight
            
        MERGE (c:Category {name: $category})
        MERGE (pk)-[:BELONGS_TO_CATEGORY]->(c)
        """
        graph.query(cypher, params=p)
        
        for victim in p['preys_on']:
            cypher_rel = """
            MATCH (pred:Pokemon {name: $pred_name})
            MERGE (prey:Pokemon {name: $prey_name})
            MERGE (pred)-[:PREYS_ON]->(prey)
            """
            graph.query(cypher_rel, params={"pred_name": p['name'], "prey_name": victim})

    logger.info("Database seeding complete!")
    graph.refresh_schema()
    logger.info("Graph schema updated.")

if __name__ == "__main__":
    pdf_path = "C:\\Users\\harsh\\Desktop\\Langchain_projects\\Graph RAG System\\pokedex.pdf"
    data = parse_pokedex(pdf_path)
    logger.info(f"Extracted {len(data)} entries.")
    if data:
        # Dump a sample to visually check parsing
        sample = next((d for d in data if 'Pidgeot' in d['name']), data[0])
        logger.info(f"Sample parsed data:\n{sample}")
        seed_graph(data)
