from neo4j import GraphDatabase
from .config_loader import load_config

cfg = load_config("config/vector_qwen.yaml")
graph_cfg = cfg["graph"]

uri = graph_cfg["neo4j_uri"]
user = graph_cfg["neo4j_user"]
password = graph_cfg["neo4j_password"]

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    result = session.run("RETURN 'connected to Neo4j Aura' AS msg")
    print(result.single()["msg"])
