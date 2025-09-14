from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class KYCDecision:
    decision: str
    risk_score: int
    reasoning: List[str]
    evidence: List[str]
    next_steps: List[str]


class HumanKYCAgent:
    def __init__(self, agent_name: str, expertise: str):
        self.agent_name = agent_name
        self.expertise = expertise

    def log_step(self, step: str, finding: str, evidence: Optional[str] = None) -> str:
        log_entry = f"[{self.agent_name}] {step}: {finding}"
        if evidence:
            log_entry += f" (Evidence: {evidence})"
        return log_entry


class NameScreeningAgent(HumanKYCAgent):
    def __init__(self):
        super().__init__("Name Screening Officer", "Sanctions list checking")

    def screen_entity(self, entity_name: str, nationality: Optional[str] = None) -> Dict[str, Any]:
        reasoning = []
        evidence = []
        risk_factors = []

        reasoning.append(self.log_step("Initial Name Review", f"Analyzing '{entity_name}' for sanctions risk indicators"))

        high_risk_indicators = self._check_name_patterns(entity_name)
        if high_risk_indicators:
            reasoning.append(self.log_step("Name Pattern Analysis", f"Found {len(high_risk_indicators)} risk indicators", ", ".join(high_risk_indicators)))
            risk_factors.extend(high_risk_indicators)
            evidence.append(f"Name patterns: {', '.join(high_risk_indicators)}")

        if nationality:
            nationality_risk = self._assess_nationality_risk(nationality)
            if nationality_risk['risk_level'] > 5:
                reasoning.append(self.log_step("Nationality Assessment", f"High-risk nationality detected: {nationality}", f"Risk level: {nationality_risk['risk_level']}/10"))
                risk_factors.append(f"High-risk nationality: {nationality}")
                evidence.append(f"Nationality risk: {nationality} (Level {nationality_risk['risk_level']})")

        matches_found = len(risk_factors) > 0
        conclusion = "Entity flagged for review based on {len(risk_factors)} risk factors" if matches_found else "No immediate sanctions risk indicators identified"
        reasoning.append(self.log_step("Risk Assessment Conclusion", conclusion))

        return {
            "agent": self.agent_name,
            "entity_screened": entity_name,
            "matches_found": matches_found,
            "risk_factors": risk_factors,
            "evidence": evidence,
            "reasoning": reasoning,
            "human_decision": "FLAG_FOR_REVIEW" if matches_found else "CLEAR_TO_PROCEED"
        }

    def _check_name_patterns(self, name: str) -> List[str]:
        indicators = []
        name_lower = name.lower()
        patterns = [
            (r'\bzorax\b', "High-risk defense entity pattern"),
            (r'\bzorax\s+defense\b', "Defense industry entity (enhanced screening required)"),
            (r'\bnorthstar\b', "Restricted trading entity pattern"),
            (r'\bmiddle\s+east\b', "Middle Eastern entity designation"),
            (r'\bdefense|military\b', "Defense/military organization designation (enhanced screening required)"),
            (r'\btrading|import|export\b', "Trade-related business (requires enhanced due diligence)"),
            (r'\benergy|petroleum|oil\b', "Energy sector entity (enhanced due diligence required)"),
            (r'\bmining|mineral\b', "Mining sector entity (enhanced due diligence required)"),
            (r'\btech|technology\b', "Technology sector entity")
        ]
        for pattern, indicator in patterns:
            if re.search(pattern, name_lower):
                indicators.append(indicator)
        return indicators

    def _assess_nationality_risk(self, nationality: str) -> Dict[str, Any]:
        risk_levels = {
            'SY': (10, 'Syria - OFAC sanctioned jurisdiction'),
            'IR': (10, 'Iran - comprehensive sanctions regime'),
            'KP': (10, 'North Korea - comprehensive sanctions'),
            'AF': (9, 'Afghanistan - Taliban-controlled territory'),
            'RU': (8, 'Russia - sectoral sanctions'),
            'BY': (7, 'Belarus - targeted sanctions'),
            'MM': (7, 'Myanmar - military junta sanctions'),
            'GB': (1, 'United Kingdom - low risk jurisdiction'),
            'US': (1, 'United States - low risk jurisdiction'),
            'CA': (1, 'Canada - low risk jurisdiction'),
        }
        level, reason = risk_levels.get(nationality, (5, f'{nationality} - standard risk assessment required'))
        return {'nationality': nationality, 'risk_level': level, 'reasoning': reason}


class BeneficialOwnershipAgent(HumanKYCAgent):
    def __init__(self):
        super().__init__("Beneficial Ownership Officer", "UBO identification and risk assessment")

    def analyze_ownership(self, business_name: str, beneficial_owners: List[Any]) -> Dict[str, Any]:
        reasoning = []
        evidence = []
        risk_flags = []

        reasoning.append(self.log_step("UBO Identification", f"Identified {len(beneficial_owners)} beneficial owner(s) for {business_name}"))

        for i, owner in enumerate(beneficial_owners, 1):
            owner_risk = self._assess_individual_risk(owner)
            reasoning.append(self.log_step(f"UBO {i} Assessment", f"Risk level: {owner_risk['risk_level']}/10 for {owner.name}", owner_risk['reasoning']))
            if owner_risk['risk_level'] >= 7:
                risk_flags.append(f"High-risk UBO: {owner.name} ({owner_risk['reasoning']})")
                evidence.append(f"UBO {i}: {owner.name} - {owner_risk['reasoning']}")

        if not beneficial_owners:
            reasoning.append(self.log_step("Ownership Structure Warning", "No beneficial owners disclosed - requires further investigation"))
            risk_flags.append("Missing UBO information")

        return {
            "agent": self.agent_name,
            "business_analyzed": business_name,
            "ubo_count": len(beneficial_owners),
            "high_risk_ubos": sum(1 for o in beneficial_owners if self._assess_individual_risk(o)['risk_level'] >= 7),
            "risk_flags": risk_flags,
            "evidence": evidence,
            "reasoning": reasoning,
            "compliance_status": "ENHANCED_DUE_DILIGENCE_REQUIRED" if risk_flags else "STANDARD_MONITORING"
        }

    def _assess_individual_risk(self, person: Any) -> Dict[str, Any]:
        """Assess individual risk like a human compliance officer"""
        risk_score = 1
        reasons = []

        # Nationality risk (standard KYC factor)
        nationality_risk = {
            'SY': 10, 'IR': 10, 'KP': 10, 'AF': 9, 'RU': 8,
            'GB': 1, 'US': 1, 'CA': 1, 'AU': 1, 'DE': 2
        }
        nat_risk = nationality_risk.get(person.nationality, 5)
        if nat_risk >= 7:
            reasons.append(f"High-risk nationality: {person.nationality}")
            risk_score = max(risk_score, nat_risk)

        # Name-based risk assessment
        if hasattr(person, 'name'):
            if re.search(r'\bal[-\s]assad', person.name.lower()):
                reasons.append("Connection to Syrian political family")
                risk_score = 10
            elif re.search(r'\bal[-\s]', person.name.lower()):
                reasons.append("Arabic naming pattern - enhanced screening required")
                risk_score = max(risk_score, 6)

        # Current country vs nationality mismatch
        if hasattr(person, 'current_country') and person.current_country != person.nationality:
            if person.nationality in ['SY', 'IR', 'AF']:
                reasons.append(f"High-risk national residing in {person.current_country}")
                risk_score = max(risk_score, 8)

        return {
            'risk_level': risk_score,
            'reasoning': '; '.join(reasons) if reasons else 'Standard risk profile'
        }


class ComplianceDecisionAgent(HumanKYCAgent):
    """Agent that makes final compliance decisions like a senior compliance officer"""

    def __init__(self):
        super().__init__("Senior Compliance Officer", "Final KYC decision making")

    def make_decision(self, screening_results: Dict, ownership_results: Dict) -> KYCDecision:
        """Make final KYC decision following human decision-making process"""
        reasoning = []
        evidence = []
        next_steps = []

        # Step 1: Review all agent findings
        reasoning.append(self.log_step(
            "Case Review Initiation",
            "Reviewing findings from screening and ownership analysis"
        ))

        # Step 2: Risk Aggregation
        total_risk_factors = (len(screening_results.get('risk_factors', [])) +
                            len(ownership_results.get('risk_flags', [])))

        high_risk_ubos = ownership_results.get('high_risk_ubos', 0)
        sanctions_flags = screening_results.get('matches_found', False)

        reasoning.append(self.log_step(
            "Risk Factor Analysis",
            f"Total risk factors: {total_risk_factors}, High-risk UBOs: {high_risk_ubos}, Sanctions flags: {sanctions_flags}"
        ))

        # Step 3: Decision Matrix (following standard KYC decision tree)
        if sanctions_flags and high_risk_ubos > 0:
            decision = "REJECT"
            risk_score = 10
            reasoning.append(self.log_step(
                "Decision: REJECT",
                "Multiple high-risk indicators - sanctions exposure and high-risk beneficial ownership"
            ))
            next_steps = ["File SAR (Suspicious Activity Report)", "Document rejection reason", "Notify senior management"]

        elif sanctions_flags or high_risk_ubos > 0:
            decision = "REVIEW"
            risk_score = 8
            reasoning.append(self.log_step(
                "Decision: REVIEW",
                "Significant risk indicators require enhanced due diligence"
            ))
            next_steps = ["Conduct enhanced due diligence", "Request additional documentation", "Senior officer review"]

        elif total_risk_factors > 2:
            decision = "REVIEW"
            risk_score = 6
            reasoning.append(self.log_step(
                "Decision: REVIEW",
                "Multiple moderate risk factors warrant additional scrutiny"
            ))
            next_steps = ["Standard enhanced monitoring", "Periodic review in 6 months"]

        else:
            decision = "ACCEPT"
            risk_score = max(2, total_risk_factors)
            reasoning.append(self.log_step(
                "Decision: ACCEPT",
                "Risk factors within acceptable parameters for onboarding"
            ))
            next_steps = ["Standard ongoing monitoring", "Annual review"]

        # Step 4: Evidence compilation
        evidence.extend(screening_results.get('evidence', []))
        evidence.extend(ownership_results.get('evidence', []))

        reasoning.append(self.log_step(
            "Case Documentation",
            f"Decision documented with {len(evidence)} pieces of supporting evidence"
        ))

        return KYCDecision(
            decision=decision,
            risk_score=risk_score,
            reasoning=reasoning,
            evidence=evidence,
            next_steps=next_steps
        )