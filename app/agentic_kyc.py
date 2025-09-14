from datetime import datetime
from typing import Dict, List, Any
import yaml
from pydantic import BaseModel, Field, validator, ConfigDict

from .kyc import KYCInput


class AgentDecision(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    agent_name: str = Field(...)
    decision: str = Field(...)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    timestamp: str = Field(...)
    processing_time_ms: int = Field(default=0, ge=0)

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow().isoformat()


class AuditLog(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    case_id: str = Field(...)
    input_data: Dict[str, Any] = Field(...)
    agent_decisions: List[AgentDecision] = Field(default_factory=list)
    final_decision: str = Field(...)
    final_reasoning: List[str] = Field(default_factory=list)
    lineage: List[Dict[str, Any]] = Field(default_factory=list)
    version_info: Dict[str, str] = Field(...)
    processing_time_total_ms: int = Field(ge=0)
    created_at: str = Field(...)

    def to_yaml(self) -> str:
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


def create_audit_log_from_result(kyc_result: Dict[str, Any], input_data: KYCInput) -> AuditLog:
    return AuditLog(
        case_id=kyc_result["case_id"],
        input_data=input_data.model_dump(),
        agent_decisions=[],
        final_decision=kyc_result["verdict"],
        final_reasoning=kyc_result["audit_trail"],
        lineage=[],
        version_info={"version": "1.0.0"},
        processing_time_total_ms=0,
        created_at=datetime.utcnow().isoformat()
    )