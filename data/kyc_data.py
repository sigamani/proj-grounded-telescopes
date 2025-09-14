FALSE_POSITIVE_CASES = [
    {
        "case_id": "FP001",
        "description": "Common name collision - different Michael Johnsons",
        "business_name": "JOHNSON ELECTRICAL SERVICES LIMITED",
        "address": "12 Industrial Park Way, Manchester",
        "country_code": "GB",
        "beneficial_owners": [
            {
                "id": 1,
                "name": "JOHNSON, Michael Robert",
                "nationality": "GB",
                "current_country": "GB",
                "dob": "1976-07-15",
            }
        ],
        "expected_verdict": "ACCEPT",
        "expected_risk": 2,
        "sanctions_context": "Common name",
    },
    {
        "case_id": "FP002",
        "description": "Similar names, different contexts",
        "business_name": "BROWN MEDICAL CONSULTING LIMITED",
        "address": "45 Royal Mile, Edinburgh, EH1 2PB",
        "country_code": "GB",
        "beneficial_owners": [
            {
                "id": 1,
                "name": "BROWN, Sarah Jane",
                "nationality": "GB",
                "current_country": "GB",
                "dob": "1979-11-20",
            }
        ],
        "expected_verdict": "ACCEPT",
        "expected_risk": 2,
        "sanctions_context": "Different person",
    },
]

TRUE_POSITIVE_CASES = [
    {
        "case_id": "TP001",
        "description": "Syrian national on sanctions",
        "business_name": "BRITISH SYRIAN CONSULTANTS LIMITED",
        "address": "45 Park Lane, London, W1K 1PN",
        "country_code": "GB",
        "beneficial_owners": [
            {
                "id": 1,
                "name": "AL-ASSAD, Asma",
                "nationality": "SY",
                "current_country": "GB",
                "dob": "1975-08-11",
            }
        ],
        "expected_verdict": "REJECT",
        "expected_risk": 10,
        "sanctions_context": "Syrian government official",
    },
    {
        "case_id": "TP002",
        "description": "Dual national on sanctions list",
        "business_name": "MIDDLE EAST TRADING COMPANY",
        "address": "Unit 12, Industrial Estate, Birmingham",
        "country_code": "GB",
        "beneficial_owners": [
            {
                "id": 1,
                "name": "AL-ZEER, Majed",
                "nationality": "GB",
                "current_country": "GB",
                "dob": "1962-08-12",
            }
        ],
        "expected_verdict": "REJECT",
        "expected_risk": 10,
        "sanctions_context": "Hamas affiliation",
    },
]

__all__ = ["TRUE_POSITIVE_CASES", "FALSE_POSITIVE_CASES"]
