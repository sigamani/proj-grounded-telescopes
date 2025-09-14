#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import neo4j
from langchain_community.graphs import Neo4jGraph


@dataclass
class SDNEntry:
    uid: str
    name: str
    sdn_type: str
    programs: List[str]
    remarks: Optional[str] = None
    addresses: Optional[List[Dict[str, str]]] = None
    alternate_names: Optional[List[str]] = None
    birth_dates: Optional[List[str]] = None
    id_numbers: Optional[List[str]] = None


@dataclass
class Address:
    uid: str
    address1: Optional[str] = None
    address2: Optional[str] = None
    address3: Optional[str] = None
    city: Optional[str] = None
    state_province: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class OFACPreprocessor:
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None

    def connect_neo4j(self):
        try:
            self.driver = neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            print("✅ Connected to Neo4j")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")

    def setup_schema(self):
        if not self.driver:
            return
        queries = [
            "CREATE CONSTRAINT sdn_uid IF NOT EXISTS FOR (s:SDN) REQUIRE s.uid IS UNIQUE",
            "CREATE INDEX sdn_name IF NOT EXISTS FOR (s:SDN) ON (s.name)"
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    def parse_ofac_xml(self, xml_file_path: str = "data/sdn.xml") -> Dict[str, List]:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        sdn_entries = []
        for sdn in root.findall('.//sdnEntry'):
            uid = sdn.get('uid')
            name = sdn.find('lastName')
            name = name.text if name is not None else uid
            programs = [p.text for p in sdn.findall('.//program') if p.text]
            sdn_entries.append(SDNEntry(uid=uid, name=name, sdn_type='Individual', programs=programs))
        return {'sdn_entries': sdn_entries, 'addresses': []}

    def store_in_neo4j(self, parsed_data: Dict[str, List]):
        if not self.driver:
            return
        with self.driver.session() as session:
            for sdn in parsed_data['sdn_entries']:
                session.run("MERGE (s:SDN {uid: $uid}) SET s.name = $name, s.sdn_type = $type",
                           uid=sdn.uid, name=sdn.name, type=sdn.sdn_type)
                for program in sdn.programs:
                    session.run("MERGE (p:Program {name: $name}) MERGE (s:SDN {uid: $uid})-[:SUBJECT_TO]->(p)",
                               name=program, uid=sdn.uid)

    def get_sanctions_matches(self, entity_name: str) -> List[Dict]:
        if not self.driver:
            return []
        with self.driver.session() as session:
            result = session.run("MATCH (s:SDN) WHERE s.name CONTAINS $name RETURN s.uid, s.name LIMIT 10",
                                name=entity_name)
            return [{'uid': r['s.uid'], 'name': r['s.name']} for r in result]

    def close(self):
        if self.driver:
            self.driver.close()


def main():
    """Main preprocessing function"""
    print("🚀 Starting OFAC Data Preprocessing for Neo4j GraphRAG")

    # Initialize preprocessor
    preprocessor = OFACPreprocessor()

    try:
        # Connect to Neo4j
        preprocessor.connect_neo4j()

        # Setup schema
        preprocessor.setup_schema()

        # Parse OFAC XML data
        xml_file = "data/sdn.xml"
        if not os.path.exists(xml_file):
            print(f"❌ OFAC XML file not found: {xml_file}")
            print("Please download the OFAC SDN list first using:")
            print('curl -L -o data/sdn.xml "https://sanctionslistservice.ofac.treas.gov/api/publicationpreview/exports/sdn.xml"')
            return

        parsed_data = preprocessor.parse_ofac_xml(xml_file)

        # Store in Neo4j
        preprocessor.store_in_neo4j(parsed_data)

        print("🎉 OFAC data preprocessing completed successfully!")

        # Test query
        print("\n🔍 Testing sanctions search...")
        test_matches = preprocessor.get_sanctions_matches("ISLAMIC")
        print(f"Found {len(test_matches)} matches for 'ISLAMIC'")
        for match in test_matches[:3]:
            print(f"  - {match['name']}")

    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        raise
    finally:
        preprocessor.close()


if __name__ == "__main__":
    main()