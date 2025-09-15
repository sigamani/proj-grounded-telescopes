from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass

import ray
import sqlite3
from pydantic import BaseModel, Field


@dataclass
class Vessel:
    id: int
    type: str
    flag_state: str
    tonnage: Optional[int] = None

@dataclass
class KYCResearchFindings:
    sanctions_matches: List[Dict]
    ownership_analysis: Dict
    graph_context: Dict
    risk_indicators: List[str]
    source_confidence: float
    business_sector: str
    sector_risk_context: str

@dataclass
class Business:
    business_name: str
    address: str
    country_code: str
    registration_id: Optional[str] = None
    website_url: Optional[str] = None
    beneficial_owners: List['Person'] = None

    def __post_init__(self):
        if self.beneficial_owners is None:
            self.beneficial_owners = []


class Person(BaseModel):
    id: int
    name: str
    nationality: str
    current_country: str
    dob: date


class KYCInput(BaseModel):
    business_name: str
    address: str
    registration_id: Optional[str] = None
    country_code: str
    website_url: Optional[str] = None
    beneficial_owners: List[Person] = Field(default_factory=list)


class KYCResult(BaseModel):
    verdict: Literal["ACCEPT", "REJECT", "REVIEW"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_score: int = Field(ge=1, le=10)
    findings: Dict[str, Any] = Field(default_factory=dict)
    audit_trail: List[str] = Field(default_factory=list)


@ray.remote
class KYCOrchestrator:
    def __init__(self):
        self.db_path = "data/watchman.db"

    def query_sanctions_db(self, query: str, params: tuple = ()) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    async def process_kyc_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        kyc_input = KYCInput(**case_data)
        context: Dict[str, Any] = {"case_id": f"KYC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
        audit_trail = []

        graph_context = self._get_graph_context(kyc_input)
        context["graph_context"] = graph_context
        audit_trail.append(f"Graph query: {graph_context['reasoning']}")

        identity_result = self._verify_identity(kyc_input)
        context.update(identity_result)
        audit_trail.append(f"Identity Verification: {identity_result['reasoning']}")

        screening_result = self._screen_sanctions(kyc_input, identity_result["entities_found"])
        context.update(screening_result)
        audit_trail.append(f"Sanctions Screening: {screening_result['reasoning']}")

        human_decision = self._make_decision(screening_result, identity_result.get('ownership_analysis', {}))
        final_verdict = human_decision.decision
        risk_score = human_decision.risk_score
        audit_trail.extend([f"Human Decision Process: {reason}" for reason in human_decision.reasoning[-3:]])

        context["human_evidence"] = human_decision.evidence
        context["next_steps"] = human_decision.next_steps

        end_time = datetime.now()
        processing_time = f"{(end_time - start_time).total_seconds():.2f}s"
        confidence = min(95, max(60, 100 - (risk_score * 8)))

        return {
            "case_id": context["case_id"],
            "verdict": final_verdict,
            "risk_score": risk_score,
            "reasoning": " | ".join(audit_trail),
            "audit_trail": audit_trail,
            "confidence": confidence,
            "processing_time": processing_time,
        }

    def _get_graph_context(self, kyc_input: KYCInput) -> Dict[str, Any]:
        return {"reasoning": "Mock graph context", "connections": []}

    def _verify_identity(self, kyc_input: KYCInput) -> Dict[str, Any]:
        entities = [kyc_input.business_name] + [owner.name for owner in kyc_input.beneficial_owners]
        from .human_kyc_process import BeneficialOwnershipAgent
        human_agent = BeneficialOwnershipAgent()
        ownership_analysis = human_agent.analyze_ownership(kyc_input.business_name, kyc_input.beneficial_owners)
        return {
            "entities_found": entities,
            "beneficial_owners": entities[1:],
            "ownership_analysis": ownership_analysis,
            "reasoning": f"Identity verification for {kyc_input.business_name} with {len(entities)-1} beneficial owners",
            "human_process_log": ownership_analysis.get('reasoning', [])
        }

    def _screen_sanctions(self, kyc_input: KYCInput, entities: List[str]) -> Dict[str, Any]:
        from .human_kyc_process import NameScreeningAgent
        screening_agent = NameScreeningAgent()
        all_screening_results = []
        all_risk_factors = []
        human_reasoning = []

        for entity in entities:
            nationality = kyc_input.country_code
            for owner in kyc_input.beneficial_owners:
                if owner.name == entity:
                    nationality = owner.nationality
                    break
            screening_result = screening_agent.screen_entity(entity, nationality)
            all_screening_results.append(screening_result)
            all_risk_factors.extend(screening_result.get('risk_factors', []))
            human_reasoning.extend(screening_result.get('reasoning', []))

        matches_found = sum(1 for result in all_screening_results if result.get('matches_found', False))
        final_reasoning = f"Human-process screening: {matches_found} entities flagged for review"
        if human_reasoning:
            final_reasoning += " | Human officer steps: " + " | ".join(human_reasoning[-3:])

        return {
            "screening_results": all_screening_results,
            "entities_flagged": matches_found,
            "risk_factors": all_risk_factors,
            "reasoning": final_reasoning,
            "human_process_log": human_reasoning,
        }

    def _make_decision(self, screening_result: Dict, ownership_analysis: Dict) -> Any:
        from .human_kyc_process import ComplianceDecisionAgent
        decision_agent = ComplianceDecisionAgent()
        return decision_agent.make_decision(screening_result, ownership_analysis)


def create_kyc_batch_processor():
    return {
        "processor_type": "cpu_fallback",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size": 8,
        "max_tokens": 512,
        "temperature": 0.1,
        "status": "active"
    }
