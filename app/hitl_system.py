#!/usr/bin/env python3
"""
Human-in-the-Loop (HITL) Integration System
Streams borderline cases to human reviewers with explainable decision artifacts
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

from pydantic import BaseModel


class ReviewStatus(Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


@dataclass
class HumanReviewCase:
    """Case queued for human review"""
    case_id: str
    priority: int  # 1-5, 5 being highest
    review_type: str  # "borderline", "high_risk", "policy_exception"
    business_name: str
    risk_score: int
    ai_decision: str
    ai_confidence: float
    evidence_package: Dict[str, Any]
    reviewer_instructions: List[str]
    created_at: str
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_reviewer: Optional[str] = None
    human_decision: Optional[str] = None
    human_reasoning: Optional[str] = None
    review_completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HITLQueue:
    """Human review queue management"""

    def __init__(self, queue_directory: str = "data/hitl_queue"):
        self.queue_dir = Path(queue_directory)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.pending_file = self.queue_dir / "pending_reviews.json"
        self.completed_file = self.queue_dir / "completed_reviews.json"

    def should_flag_for_review(self, audit_log) -> tuple[bool, List[str]]:
        """Determine if case should be flagged for human review"""

        # Rule-based HITL triggers
        final_decision = audit_log.final_decision
        risk_score = 0
        confidence = 0.0

        # Extract metrics from agent decisions
        for decision in audit_log.agent_decisions:
            if decision.agent_name == "Adjudication":
                confidence = decision.confidence
                for evidence in decision.evidence:
                    risk_score = evidence.get("risk_score", 0)
                break

        # HITL Trigger Rules
        triggers = []

        # 1. Borderline cases with low confidence
        if 0.6 < confidence < 0.85:
            triggers.append("Low AI confidence")

        # 2. High-risk cases with REVIEW decision
        if final_decision == "REVIEW" and risk_score >= 7:
            triggers.append("High-risk review case")

        # 3. Policy edge cases
        if final_decision == "ACCEPT" and risk_score >= 6:
            triggers.append("Risk/decision mismatch")

        # 4. Sanctions matches requiring verification
        sanctions_detected = any(
            len(d.risk_factors) > 0 and "sanctions" in str(d.risk_factors).lower()
            for d in audit_log.agent_decisions
        )
        if sanctions_detected:
            triggers.append("Sanctions screening verification required")

        return len(triggers) > 0, triggers

    def queue_for_review(self, audit_log, triggers: List[str]) -> HumanReviewCase:
        """Queue a case for human review"""

        # Extract case details
        final_decision_agent = next(
            (d for d in audit_log.agent_decisions if d.agent_name == "Adjudication"),
            None
        )

        risk_score = 5
        confidence = 0.5
        if final_decision_agent:
            confidence = final_decision_agent.confidence
            for evidence in final_decision_agent.evidence:
                risk_score = evidence.get("risk_score", 5)

        # Determine priority
        priority = self._calculate_priority(risk_score, confidence, triggers)

        # Create evidence package
        evidence_package = {
            "full_audit_log": audit_log.to_dict(),
            "key_findings": [
                f"{d.agent_name}: {d.decision} (conf: {d.confidence:.2f})"
                for d in audit_log.agent_decisions
            ],
            "risk_factors_summary": self._extract_risk_factors(audit_log),
            "ai_reasoning_chain": [
                reason for decision in audit_log.agent_decisions
                for reason in decision.reasoning
            ],
            "triggers": triggers
        }

        # Create reviewer instructions
        instructions = self._generate_reviewer_instructions(
            audit_log.final_decision, risk_score, triggers
        )

        # Create review case
        review_case = HumanReviewCase(
            case_id=audit_log.case_id,
            priority=priority,
            review_type=self._determine_review_type(triggers),
            business_name=audit_log.input_data["business_name"],
            risk_score=risk_score,
            ai_decision=audit_log.final_decision,
            ai_confidence=confidence,
            evidence_package=evidence_package,
            reviewer_instructions=instructions,
            created_at=datetime.utcnow().isoformat()
        )

        # Save to queue
        self._save_to_queue(review_case)

        return review_case

    def _calculate_priority(self, risk_score: int, confidence: float, triggers: List[str]) -> int:
        """Calculate review priority (1-5, 5 highest)"""
        priority = 1

        # Risk-based priority
        if risk_score >= 8:
            priority = 5
        elif risk_score >= 6:
            priority = 4
        elif risk_score >= 4:
            priority = 3
        else:
            priority = 2

        # Confidence adjustment
        if confidence < 0.7:
            priority = min(5, priority + 1)

        # Trigger-based adjustments
        high_priority_triggers = ["sanctions", "high-risk", "policy"]
        if any(trigger.lower() in t.lower() for trigger in high_priority_triggers for t in triggers):
            priority = 5

        return priority

    def _determine_review_type(self, triggers: List[str]) -> str:
        """Determine the type of human review needed"""
        trigger_text = " ".join(triggers).lower()

        if "sanctions" in trigger_text:
            return "sanctions_verification"
        elif "high-risk" in trigger_text:
            return "high_risk_assessment"
        elif "policy" in trigger_text or "mismatch" in trigger_text:
            return "policy_exception"
        else:
            return "borderline_case"

    def _extract_risk_factors(self, audit_log) -> List[str]:
        """Extract and summarize risk factors"""
        all_risk_factors = []
        for decision in audit_log.agent_decisions:
            all_risk_factors.extend(decision.risk_factors)

        # Deduplicate and categorize
        unique_factors = list(set(all_risk_factors))
        return unique_factors[:10]  # Top 10 most important

    def _generate_reviewer_instructions(self, ai_decision: str, risk_score: int, triggers: List[str]) -> List[str]:
        """Generate human-readable instructions for reviewers"""
        instructions = [
            f"Review this KYC case with AI decision: {ai_decision}",
            f"Current risk score: {risk_score}/10",
            f"Review triggered by: {', '.join(triggers)}"
        ]

        # Specific guidance based on triggers
        trigger_text = " ".join(triggers).lower()

        if "sanctions" in trigger_text:
            instructions.extend([
                "⚠️  SANCTIONS REVIEW REQUIRED:",
                "1. Verify name matches against official OFAC/EU/UN lists",
                "2. Check for false positives due to common names",
                "3. Assess business context and geographic risk factors",
                "4. Document any manual screening performed"
            ])

        if "high-risk" in trigger_text:
            instructions.extend([
                "🔍 HIGH-RISK CASE ASSESSMENT:",
                "1. Review beneficial ownership structure",
                "2. Assess geographic and sector risk factors",
                "3. Consider regulatory reporting requirements",
                "4. Evaluate need for enhanced due diligence"
            ])

        if "confidence" in trigger_text:
            instructions.extend([
                "🤖 LOW AI CONFIDENCE - HUMAN JUDGMENT NEEDED:",
                "1. Review AI reasoning chain for gaps",
                "2. Apply professional compliance experience",
                "3. Consider additional data sources if needed",
                "4. Document rationale for final decision"
            ])

        instructions.append("Final Decision Options: ACCEPT, REJECT, REQUEST_MORE_INFO")
        return instructions

    def _save_to_queue(self, review_case: HumanReviewCase):
        """Save review case to pending queue"""
        # Load existing queue
        pending_cases = []
        if self.pending_file.exists():
            with open(self.pending_file, 'r') as f:
                pending_cases = json.load(f)

        # Add new case
        pending_cases.append(review_case.to_dict())

        # Sort by priority (highest first)
        pending_cases.sort(key=lambda x: (-x['priority'], x['created_at']))

        # Save updated queue
        with open(self.pending_file, 'w') as f:
            json.dump(pending_cases, f, indent=2, default=str)

    def get_pending_reviews(self, limit: int = 20) -> List[HumanReviewCase]:
        """Get pending review cases for human reviewers"""
        if not self.pending_file.exists():
            return []

        with open(self.pending_file, 'r') as f:
            cases_data = json.load(f)

        cases = []
        for case_data in cases_data[:limit]:
            case_data['status'] = ReviewStatus(case_data['status'])
            cases.append(HumanReviewCase(**case_data))

        return cases

    def submit_human_decision(self, case_id: str, reviewer: str, decision: str, reasoning: str):
        """Submit human reviewer decision"""
        # Load pending cases
        if not self.pending_file.exists():
            return False

        with open(self.pending_file, 'r') as f:
            pending_cases = json.load(f)

        # Find and update case
        case_found = False
        for case in pending_cases:
            if case['case_id'] == case_id:
                case['status'] = ReviewStatus.APPROVED.value if decision == "ACCEPT" else ReviewStatus.REJECTED.value
                case['assigned_reviewer'] = reviewer
                case['human_decision'] = decision
                case['human_reasoning'] = reasoning
                case['review_completed_at'] = datetime.utcnow().isoformat()
                case_found = True
                break

        if not case_found:
            return False

        # Move to completed queue
        completed_cases = []
        if self.completed_file.exists():
            with open(self.completed_file, 'r') as f:
                completed_cases = json.load(f)

        # Remove from pending
        completed_case = next(c for c in pending_cases if c['case_id'] == case_id)
        pending_cases = [c for c in pending_cases if c['case_id'] != case_id]

        # Add to completed
        completed_cases.append(completed_case)

        # Save both files
        with open(self.pending_file, 'w') as f:
            json.dump(pending_cases, f, indent=2, default=str)

        with open(self.completed_file, 'w') as f:
            json.dump(completed_cases, f, indent=2, default=str)

        return True


class SyntheticTrainingDataGenerator:
    """Generate safe training data for fine-tuning without PII exposure"""

    def __init__(self):
        self.templates = {
            "low_risk": [
                ("ACME CONSULTING LIMITED", "London", "GB", [], "Standard consulting firm"),
                ("TECH INNOVATIONS INC", "Toronto", "CA", [], "Technology company"),
                ("GREEN ENERGY SOLUTIONS", "Berlin", "DE", [], "Renewable energy business")
            ],
            "medium_risk": [
                ("INTERNATIONAL TRADING CO", "Dubai", "AE", [("OWNER1", "AE")], "Cross-border trading"),
                ("GLOBAL INVESTMENT FUND", "Singapore", "SG", [("OWNER2", "SG")], "Investment management"),
                ("IMPORT EXPORT SERVICES", "Istanbul", "TR", [("OWNER3", "TR")], "Trade facilitation")
            ],
            "high_risk": [
                ("MIDDLE EASTERN ENTERPRISES", "Baghdad", "IQ", [("AL-MANAGER", "IQ")], "Regional conglomerate"),
                ("REVOLUTIONARY TRADING", "Tehran", "IR", [("CONTROLLER", "IR")], "State-linked entity"),
                ("SANCTIONS EVASION LLC", "Minsk", "BY", [("OLIGARCH", "RU")], "Shell company pattern")
            ]
        }

    def generate_training_cases(self, count_per_category: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic training cases for model fine-tuning"""
        training_data = []

        for risk_level, templates in self.templates.items():
            for i in range(count_per_category):
                template = templates[i % len(templates)]

                # Generate variations
                case = {
                    "input": {
                        "business_name": self._vary_name(template[0]),
                        "address": f"{i+1} {template[1]} Street, {template[1]}",
                        "country_code": template[2],
                        "beneficial_owners": [
                            {"name": owner[0], "nationality": owner[1]}
                            for owner in template[3]
                        ]
                    },
                    "expected_output": {
                        "risk_level": risk_level,
                        "decision": self._map_risk_to_decision(risk_level),
                        "reasoning": template[4],
                        "confidence": self._generate_confidence(risk_level)
                    },
                    "metadata": {
                        "synthetic": True,
                        "template_id": templates.index(template),
                        "variation": i,
                        "generated_at": datetime.utcnow().isoformat()
                    }
                }

                training_data.append(case)

        return training_data

    def _vary_name(self, base_name: str) -> str:
        """Create variations of entity names"""
        variations = {
            "CONSULTING": ["ADVISORY", "SERVICES", "SOLUTIONS"],
            "TRADING": ["COMMERCE", "BUSINESS", "ENTERPRISES"],
            "INTERNATIONAL": ["GLOBAL", "WORLDWIDE", "UNIVERSAL"],
            "LIMITED": ["LTD", "LLC", "CORP"]
        }

        result = base_name
        for original, replacements in variations.items():
            if original in result:
                import random
                replacement = random.choice(replacements)
                result = result.replace(original, replacement, 1)
                break

        return result

    def _map_risk_to_decision(self, risk_level: str) -> str:
        """Map risk level to expected decision"""
        mapping = {
            "low_risk": "ACCEPT",
            "medium_risk": "REVIEW",
            "high_risk": "REJECT"
        }
        return mapping.get(risk_level, "REVIEW")

    def _generate_confidence(self, risk_level: str) -> float:
        """Generate appropriate confidence scores"""
        import random
        ranges = {
            "low_risk": (0.85, 0.95),
            "medium_risk": (0.70, 0.85),
            "high_risk": (0.90, 0.99)
        }
        min_conf, max_conf = ranges.get(risk_level, (0.75, 0.85))
        return round(random.uniform(min_conf, max_conf), 2)


