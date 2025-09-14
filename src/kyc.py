from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

import ray
from pydantic import BaseModel, Field, constr

ISO2 = constr(strict=True, min_length=2, max_length=2)


class Person(BaseModel):
    id: int
    name: constr(strip_whitespace=True, min_length=1)
    nationality: ISO2
    current_country: ISO2
    dob: date


class KYCInput(BaseModel):
    business_name: constr(strip_whitespace=True, min_length=1)
    address: constr(strip_whitespace=True, min_length=1)
    registration_id: Optional[str] = None
    country_code: ISO2
    website_url: Optional[str] = None
    beneficial_owners: List[Person] = Field(default_factory=list)


class KYCResult(BaseModel):
    verdict: Literal["ACCEPT", "REJECT", "REVIEW"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_score: int = Field(ge=1, le=10)
    findings: Dict[str, Any] = Field(default_factory=dict)
    audit_trail: List[str] = Field(default_factory=list)


class GraphRAGKnowledgeBase:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_sanctions_context(
        self, entity_name: str, nationality: str
    ) -> Dict[str, Any]:
        return {
            "matches": [],
            "risk_factors": [],
            "contextual_reasoning": f"Analyzing {entity_name} ({nationality})"
        }

    def assess_country_risk(self, country_code: str) -> Dict[str, Any]:
        risk_scores = {"GB": 1, "US": 1, "RU": 9, "CN": 7, "IR": 10}
        base_risk = risk_scores.get(country_code, 5)
        return {
            "country_risk_score": base_risk,
            "risk_factors": [f"Country code {country_code} base risk assessment"],
            "reasoning": f"Standard risk assessment for {country_code}",
        }


class KYCAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.db_path = "data/watchman.db"

    @abstractmethod
    async def process(
        self, input_data: KYCInput, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        pass

    def query_sanctions_db(self, query: str, params: tuple = ()) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(query, params).fetchall()]


class IdentityAgent(KYCAgent):
    async def process(
        self, input_data: KYCInput, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        entities = [input_data.business_name]
        beneficial_owner_names = [owner.name for owner in input_data.beneficial_owners]
        entities.extend(beneficial_owner_names)
        return {
            "agent": self.name,
            "entities_found": entities,
            "beneficial_owners": beneficial_owner_names,
            "reasoning": f"{input_data.business_name}"
        }


class ScreeningAgent(KYCAgent):
    async def process(
        self, input_data: KYCInput, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        matches = []
        entities = (
            context.get("entities_found", []) if context else [input_data.business_name]
        )
        for entity in entities:
            sdn_matches = self.query_sanctions_db(
                """SELECT s.sdn_name, s.programs, s.remarks, s.sdn_type,
                          a.country, alt.alternate_name
                   FROM sdn s
                   LEFT JOIN addresses a ON s.entity_id = a.entity_id
                   LEFT JOIN alt_names alt ON s.entity_id = alt.entity_id
                   WHERE s.sdn_name LIKE ? OR s.sdn_name LIKE ?
                      OR alt.alternate_name LIKE ?
                      OR (a.country = ? AND s.sdn_name LIKE ?)""",
                (
                    f"%{entity}%",
                    f"%{entity.replace(' ', '%')}%",
                    f"%{entity}%",
                    input_data.country_code,
                    f"%{entity.split()[0]}%" if entity.split() else f"%{entity}%",
                ),
            )
            matches.extend(sdn_matches)
        reasoning = (
            f"GraphRAG screening found {len(matches)} contextual sanctions matches. "
        )
        if matches:
            programs = set(
                match.get("programs", "") for match in matches if match.get("programs")
            )
            reasoning += f"Programs: {', '.join(programs)}."
        else:
            reasoning += "No sanctions matches found."
        return {
            "agent": self.name,
            "sanctions_matches": matches,
            "reasoning": reasoning,
        }


@ray.remote
class KYCOrchestrator:
    def __init__(self):
        self.identity_agent = IdentityAgent("identity-verification")
        self.screening_agent = ScreeningAgent("sanctions-screening")

    async def process_kyc_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        kyc_input = KYCInput(**case_data)
        context = {"case_id": f"KYC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
        audit_trail = []

        identity_result = await self.identity_agent.process(kyc_input, context)
        context.update(identity_result)
        audit_trail.append(f"Identity Verification: {identity_result['reasoning']}")

        final_verdict = (
            "REJECT"
            if context.get("sanctions_hit") or context.get("high_risk")
            else "APPROVE"
        )
        end_time = datetime.now()
        processing_time = f"{(end_time - start_time).total_seconds():.2f}s"
        confidence = min(95, max(60, 100 - (context.get("risk_score", 5) * 8)))

        return {
            "case_id": context["case_id"],
            "verdict": final_verdict,
            "risk_score": context.get("risk_score", 5),
            "reasoning": " | ".join(audit_trail),
            "audit_trail": audit_trail,
            "confidence": confidence,
            "processing_time": processing_time,
        }


def create_kyc_batch_processor():
    return {
        "processor_type": "vllm_ray_batch",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size": 32,
        "max_tokens": 512,
        "temperature": 0.1,
    }
