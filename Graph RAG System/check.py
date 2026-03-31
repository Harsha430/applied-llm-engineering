import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv('C:\\Users\\harsh\\Desktop\\Langchain_projects\\Graph RAG System\\.env')
g = Neo4jGraph()

query = "MATCH (p:Pokemon) WHERE p.name CONTAINS 'Pidgeot' RETURN p.name as name, length(p.name) as len"
res = g.query(query)
print(f"Results: {res}")
if res:
    name = res[0]['name']
    print(f"Repr: {repr(name)}")
    print(f"Length: {len(name)}")
