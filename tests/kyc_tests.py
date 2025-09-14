import ray

from data.kyc_data import FALSE_POSITIVE_CASES, TRUE_POSITIVE_CASES
from app.kyc import KYCOrchestrator


def test_kyc_pipeline():
    test_case = {
        "business_name": "ISLAMIC REVOLUTIONARY GUARD",
        "address": "Tehran, Iran",
        "country_code": "IR",
        "beneficial_owners": [],
    }

    if not ray.is_initialized():
        ray.init()

    orchestrator = KYCOrchestrator.remote()
    future = orchestrator.process_kyc_case.remote(test_case)
    result = ray.get(future)

    assert result is not None
    assert "verdict" in result
    assert "audit_trail" in result


def test_enhanced_kyc_pipeline():
    if not ray.is_initialized():
        ray.init()

    high_risk_case = {
        "business_name": "ISLAMIC REVOLUTIONARY GUARD CORPS",
        "address": "Tehran, Iran",
        "country_code": "IR",
        "beneficial_owners": [],
    }

    low_risk_case = {
        "business_name": "MAPLE TECH SOLUTIONS INC",
        "address": "123 Bay Street, Toronto, ON M5J 2T3",
        "country_code": "CA",
        "beneficial_owners": [],
    }

    orchestrator = KYCOrchestrator.remote()

    high_risk_result = ray.get(orchestrator.process_kyc_case.remote(high_risk_case))
    low_risk_result = ray.get(orchestrator.process_kyc_case.remote(low_risk_case))

    assert high_risk_result["verdict"] in ["REJECT", "REVIEW"]
    assert low_risk_result["verdict"] in ["ACCEPT", "APPROVE"]


def test_batch_processing():
    test_cases = [
        {
            "business_name": "Test Company Ltd",
            "address": "123 Test Street, London, UK",
            "country_code": "GB",
            "beneficial_owners": [],
        }
    ]

    if not ray.is_initialized():
        ray.init()

    orchestrator = KYCOrchestrator.remote()
    futures = [orchestrator.process_kyc_case.remote(case) for case in test_cases]
    results = ray.get(futures)

    assert len(results) == len(test_cases)
    for result in results:
        assert "verdict" in result
        assert "risk_score" in result


def test_uk_kyc_dataset():
    if not ray.is_initialized():
        ray.init()

    orchestrator = KYCOrchestrator.remote()

    for case in TRUE_POSITIVE_CASES:
        result = ray.get(orchestrator.process_kyc_case.remote(case))
        assert result["verdict"] in ["REJECT", "REVIEW"]

    for case in FALSE_POSITIVE_CASES:
        ray.get(orchestrator.process_kyc_case.remote(case))


if __name__ == "__main__":
    test_kyc_pipeline()
    test_enhanced_kyc_pipeline()
    test_batch_processing()
    test_uk_kyc_dataset()
