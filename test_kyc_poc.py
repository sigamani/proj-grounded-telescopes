# import asyncio

import ray

from src.batch_infer import KYCOrchestrator


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

    print("KYC Processing Result:")
    print(f"Verdict: {result['verdict']}")
    print(f"Risk Score: {result['risk_score']}/10")
    print(f"Reasoning: {result['reasoning']}")
    print("\nAudit Trail:")
    for step in result["audit_trail"]:
        print(f"  - {step}")

    # Avoid ray.shutdown() in tests to prevent circular import issues
    # Ray will automatically cleanup when the process exits
    assert result is not None
    assert 'verdict' in result
    assert 'audit_trail' in result


if __name__ == "__main__":
    test_kyc_pipeline()
