# 10 comprehensive SYNTHETIC KYC test cases for end-to-end testing
# ALL DATA IS COMPLETELY FICTIONAL AND DOES NOT CONTAIN REAL PII OR ENTITY NAMES
KYC_TEST_CASES = [
    {
        "id": "TC001",
        "description": "High-risk synthetic entity from sanctioned country",
        "business_name": "ZORAX DEFENSE SYSTEMS LTD",
        "address": "742 Evergreen Terrace, Springfield, ZZ",
        "country_code": "IR",
        "expected_verdict": "REJECT",
        "expected_risk": 10
    },
    {
        "id": "TC002",
        "description": "Low-risk synthetic Canadian tech company",
        "business_name": "NOVAQUANT TECHNOLOGIES INC",
        "address": "456 Innovation Boulevard, Techville, NT M5J 2T3",
        "country_code": "CA",
        "expected_verdict": "ACCEPT",
        "expected_risk": 1
    },
    {
        "id": "TC003",
        "description": "Synthetic UK company with registration",
        "business_name": "BRITANNIA WIDGET CORPORATION PLC",
        "address": "789 Industrial Estate, Widgeton, UK",
        "registration_id": "GB12345678",
        "country_code": "GB",
        "expected_verdict": "ACCEPT",
        "expected_risk": 2
    },
    {
        "id": "TC004",
        "description": "Synthetic US company with website",
        "business_name": "PACIFIC COAST INNOVATIONS LLC",
        "address": "321 Silicon Valley Drive, Innovatown, CA 94000",
        "country_code": "US",
        "website_url": "https://pacific-coast-innovations.example",
        "expected_verdict": "ACCEPT",
        "expected_risk": 1
    },
    {
        "id": "TC005",
        "description": "High-risk synthetic entity from restricted country",
        "business_name": "NORTHSTAR TRADING COMPANY",
        "address": "123 Commerce Street, Tradetown, ZZ",
        "country_code": "KP",
        "expected_verdict": "REJECT",
        "expected_risk": 10
    },
    {
        "id": "TC006",
        "description": "Synthetic German manufacturing company",
        "business_name": "TEUTONIC ENGINEERING GMBH",
        "address": "456 Industriestrasse, Engineeringstadt, Germany",
        "country_code": "DE",
        "expected_verdict": "ACCEPT",
        "expected_risk": 2
    },
    {
        "id": "TC007",
        "description": "Medium-risk synthetic Russian energy company",
        "business_name": "URAL ENERGY HOLDINGS JSC",
        "address": "789 Energy Plaza, Energytown, Russia",
        "country_code": "RU",
        "expected_verdict": "REVIEW",
        "expected_risk": 7
    },
    {
        "id": "TC008",
        "description": "Synthetic Australian mining company",
        "business_name": "SOUTHERN CROSS MINING PTY LTD",
        "address": "321 Mining Road, Mineville, WA 6000",
        "country_code": "AU",
        "expected_verdict": "ACCEPT",
        "expected_risk": 2
    },
    {
        "id": "TC009",
        "description": "Medium-risk synthetic Venezuelan oil company",
        "business_name": "TROPICAL PETROLEUM VENEZUELA SA",
        "address": "654 Oil Avenue, Oiltown, Venezuela",
        "country_code": "VE",
        "expected_verdict": "REVIEW",
        "expected_risk": 8
    },
    {
        "id": "TC010",
        "description": "Synthetic Japanese electronics company",
        "business_name": "MOUNT FUJI ELECTRONICS CORP",
        "address": "987 Technology Street, Techcity, Tokyo",
        "country_code": "JP",
        "website_url": "https://mount-fuji-electronics.example",
        "expected_verdict": "ACCEPT",
        "expected_risk": 1
    }
]

__all__ = ["KYC_TEST_CASES"]
