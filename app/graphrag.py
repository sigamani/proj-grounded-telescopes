#!/usr/bin/env python3
from typing import Dict, List, Any, Optional
import neo4j
from langchain_community.graphs import Neo4jGraph


class GraphRAGKnowledgeBase:
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None

    def connect(self):
        try:
            self.driver = neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            print("✅ Connected to Neo4j")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            self.driver = None

    def get_sanctions_context(self, entity_name: str, nationality: Optional[str] = None) -> Dict[str, Any]:
        if not self.driver:
            return {"matches": [], "risk_factors": [], "reasoning": "Graph not available", "total_matches": 0}
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (s:SDN) WHERE toLower(s.name) CONTAINS toLower($name) RETURN s.name LIMIT 5", name=entity_name)
                matches = [r['s.name'] for r in result]
            return {"matches": matches, "risk_factors": [], "reasoning": f"Found {len(matches)} matches", "total_matches": len(matches)}
        except Exception:
            return {"matches": [], "risk_factors": [], "reasoning": "Query failed", "total_matches": 0}

    def assess_country_risk(self, country_code: str) -> Dict[str, Any]:
        if not self.driver:
            return {"country_risk_score": 5, "risk_factors": [], "reasoning": "Graph not available"}
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (c:Country {code: $code}) RETURN c.risk_score", code=country_code)
                record = result.single()
                risk_score = record['c.risk_score'] if record else 5
            return {"country_risk_score": risk_score, "risk_factors": [], "reasoning": f"Risk score: {risk_score}"}
        except Exception:
            return {"country_risk_score": 5, "risk_factors": [], "reasoning": "Query failed"}

    def close(self):
        if self.driver:
            self.driver.close()


def create_graphrag_knowledge_base(db_path: Optional[str] = None) -> GraphRAGKnowledgeBase:
    kb = GraphRAGKnowledgeBase()
    kb.connect()
    return kb