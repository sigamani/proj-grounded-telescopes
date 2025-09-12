"""
KYC Utilities with Weave Tracing Integration

Enhanced KYC processing pipeline with end-to-end tracing and evaluation.
Self-contained script for testing and exploration.
"""

import json
import os
from getpass import getpass
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import all dependencies locally to isolate functionality
try:
    import weave

    WEAVE_AVAILABLE = True
except ImportError:
    print("Weave not available - install with: pip install weave")
    WEAVE_AVAILABLE = False

    # Mock Weave functionality for testing
    class MockWeave:
        @staticmethod
        def init(project_name):
            pass

        @staticmethod
        def op():
            def decorator(func):
                return func

            return decorator

        class Evaluation:
            def __init__(self):
                pass

    weave = MockWeave()

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI not available - install with: pip install openai")
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from pydantic import BaseModel, Field, ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    print("Pydantic not available - install with: pip install pydantic>=2.6")
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None
    ValidationError = Exception

# Import from local kyc_utils if available
try:
    from kyc_utils import (
        KYCRequest,
        KYCResponse,
        sanitize_kyc_input,
        create_kyc_analysis_prompt,
        create_compliance_validation_prompt,
        validate_kyc_request,
        parse_kyc_response,
    )

    KYC_UTILS_AVAILABLE = True
except ImportError:
    print("KYC utils not available - some functionality will be limited")
    KYC_UTILS_AVAILABLE = False


# Initialize Weave project if available
if WEAVE_AVAILABLE:
    weave.init("KYC")

# Define fallback functions if dependencies not available
if not KYC_UTILS_AVAILABLE:

    def sanitize_kyc_input(row):
        """Fallback sanitization function."""
        return {
            "business_name": str(row.get("business_name", "")).strip(),
            "country_code": str(row.get("country_code", "")).upper()[:2],
            "validation_status": "PASSED" if row.get("business_name") else "FAILED",
        }

    def create_kyc_analysis_prompt(row):
        """Fallback prompt creation function."""
        return {
            "messages": [
                {"role": "system", "content": "You are a KYC analyst."},
                {
                    "role": "user",
                    "content": f"Analyze company: {row.get('business_name', 'Unknown')}",
                },
            ],
            "sampling_params": {"temperature": 0.1, "max_tokens": 1500},
        }

    def create_compliance_validation_prompt(row):
        """Fallback compliance prompt function."""
        return {
            "messages": [
                {"role": "system", "content": "You are a compliance officer."},
                {"role": "user", "content": "Validate this KYC analysis."},
            ],
            "sampling_params": {"temperature": 0.05, "max_tokens": 1000},
        }

    def validate_kyc_request(request_dict):
        """Fallback validation function."""
        required_fields = ["business_name", "address", "country_code"]
        for field in required_fields:
            if not request_dict.get(field):
                raise ValidationError(f"Missing required field: {field}")
        return type("MockRequest", (), request_dict)()


if not PYDANTIC_AVAILABLE:

    class KYCEvaluationResult:
        """Fallback evaluation result class."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return {k: v for k, v in self.__dict__.items()}


# UK Dataset Examples for Testing
UK_TEST_COMPANIES = [
    {
        "business_name": "Tesco PLC",
        "address": "Tesco House, Shire Park, Kestrel Way, Welwyn Garden City, AL7 1GA",
        "country_code": "GB",
        "registration_id": "00445790",
        "website_url": "https://www.tesco.com",
        "expected_verdict": "ACCEPT",
        "expected_risk": "Low",
        "expected_industry": "Retail",
    },
    {
        "business_name": "HSBC Holdings plc",
        "address": "8 Canada Square, London, E14 5HQ",
        "country_code": "GB",
        "registration_id": "00617987",
        "website_url": "https://www.hsbc.com",
        "expected_verdict": "REVIEW",
        "expected_risk": "Medium",
        "expected_industry": "Financial Services",
    },
    {
        "business_name": "Vodafone Group Plc",
        "address": "Vodafone House, The Connection, Newbury, Berkshire, RG14 2FN",
        "country_code": "GB",
        "registration_id": "01833679",
        "website_url": "https://www.vodafone.com",
        "expected_verdict": "ACCEPT",
        "expected_risk": "Low",
        "expected_industry": "Telecommunications",
    },
    {
        "business_name": "BP p.l.c.",
        "address": "1 St James's Square, London, SW1Y 4PD",
        "country_code": "GB",
        "registration_id": "00102498",
        "website_url": "https://www.bp.com",
        "expected_verdict": "REVIEW",
        "expected_risk": "Medium",
        "expected_industry": "Energy",
    },
    {
        "business_name": "Unilever PLC",
        "address": "100 Victoria Embankment, London, EC4Y 0DY",
        "country_code": "GB",
        "registration_id": "00041424",
        "website_url": "https://www.unilever.com",
        "expected_verdict": "ACCEPT",
        "expected_risk": "Low",
        "expected_industry": "Consumer Goods",
    },
    {
        "business_name": "Barclays PLC",
        "address": "1 Churchill Place, London, E14 5HP",
        "country_code": "GB",
        "registration_id": "00048839",
        "website_url": "https://www.barclays.com",
        "expected_verdict": "REVIEW",
        "expected_risk": "Medium",
        "expected_industry": "Financial Services",
    },
    {
        "business_name": "Shell plc",
        "address": "Shell Mex House, Strand, London, WC2R 0DX",
        "country_code": "GB",
        "registration_id": "04366849",
        "website_url": "https://www.shell.com",
        "expected_verdict": "REVIEW",
        "expected_risk": "Medium",
        "expected_industry": "Energy",
    },
    {
        "business_name": "BT Group plc",
        "address": "BT Centre, 81 Newgate Street, London, EC1A 7AJ",
        "country_code": "GB",
        "registration_id": "04190816",
        "website_url": "https://www.bt.com",
        "expected_verdict": "ACCEPT",
        "expected_risk": "Low",
        "expected_industry": "Telecommunications",
    },
    {
        "business_name": "Rolls-Royce Holdings plc",
        "address": "62 Buckingham Gate, London, SW1E 6AT",
        "country_code": "GB",
        "registration_id": "07524813",
        "website_url": "https://www.rolls-royce.com",
        "expected_verdict": "ACCEPT",
        "expected_risk": "Low",
        "expected_industry": "Aerospace & Defense",
    },
    {
        "business_name": "AstraZeneca PLC",
        "address": "1 Francis Crick Avenue, Cambridge Biomedical Campus, Cambridge, CB2 0AA",
        "country_code": "GB",
        "registration_id": "02723534",
        "website_url": "https://www.astrazeneca.com",
        "expected_verdict": "ACCEPT",
        "expected_risk": "Low",
        "expected_industry": "Pharmaceuticals",
    },
]


class KYCEvaluationResult(BaseModel):
    """Results from KYC evaluation."""

    company_name: str
    actual_verdict: str
    expected_verdict: str
    actual_risk: str
    expected_risk: str
    actual_industry: str
    expected_industry: str
    verdict_correct: bool
    risk_correct: bool
    industry_correct: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    processing_time: float


@weave.op()
def kyc_analyze_with_llm(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform KYC analysis using OpenAI LLM with Weave tracing.

    Args:
        input_data: Sanitized KYC input data

    Returns:
        LLM analysis results
    """
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass("Enter OpenAI API key: ")

    client = OpenAI(api_key=api_key)

    # Create analysis prompt
    prompt_data = create_kyc_analysis_prompt(input_data)

    if prompt_data.get("skip_processing"):
        return {"error": "Invalid input data", "skip_processing": True}

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model for KYC analysis
            messages=prompt_data["messages"],
            **prompt_data["sampling_params"],
            response_format={"type": "json_object"},
        )

        analysis_text = response.choices[0].message.content

        # Parse JSON response
        try:
            analysis_json = json.loads(analysis_text)
        except json.JSONDecodeError:
            analysis_json = {
                "error": "Invalid JSON response",
                "raw_response": analysis_text,
            }

        return {
            "analysis_result": analysis_json,
            "raw_response": analysis_text,
            "model": "gpt-4o-mini",
            "tokens_used": response.usage.total_tokens if response.usage else 0,
            "processing_stage": "initial_analysis",
        }

    except Exception as e:
        return {
            "error": f"LLM analysis failed: {str(e)}",
            "processing_stage": "initial_analysis",
        }


@weave.op()
def kyc_compliance_validation(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform compliance validation using OpenAI LLM with Weave tracing.

    Args:
        analysis_result: Results from initial KYC analysis

    Returns:
        Compliance validation results
    """
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass("Enter OpenAI API key: ")

    client = OpenAI(api_key=api_key)

    # Create compliance validation prompt
    prompt_data = create_compliance_validation_prompt(analysis_result)

    if prompt_data.get("skip_processing"):
        return {"error": "Cannot validate invalid analysis", "skip_processing": True}

    try:
        # Call OpenAI API for compliance validation
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_data["messages"],
            **prompt_data["sampling_params"],
            response_format={"type": "json_object"},
        )

        validation_text = response.choices[0].message.content

        # Parse JSON response
        try:
            validation_json = json.loads(validation_text)
        except json.JSONDecodeError:
            validation_json = {
                "error": "Invalid JSON response",
                "raw_response": validation_text,
            }

        return {
            "validation_result": validation_json,
            "raw_response": validation_text,
            "model": "gpt-4o-mini",
            "tokens_used": response.usage.total_tokens if response.usage else 0,
            "processing_stage": "compliance_validation",
        }

    except Exception as e:
        return {
            "error": f"Compliance validation failed: {str(e)}",
            "processing_stage": "compliance_validation",
        }


@weave.op()
def process_kyc_request_traced(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process complete KYC request with full Weave tracing.

    Args:
        request_data: Raw KYC request data

    Returns:
        Complete KYC processing results
    """
    start_time = datetime.now()

    try:
        # Step 1: Validate and sanitize input
        validated_request = validate_kyc_request(request_data)
        sanitized_data = sanitize_kyc_input(validated_request.model_dump())

        if sanitized_data.get("validation_status") != "PASSED":
            return {
                "error": "Input validation failed",
                "validation_errors": sanitized_data.get("validation_errors", []),
                "processing_time": (datetime.now() - start_time).total_seconds(),
            }

        # Step 2: Initial KYC analysis with LLM
        analysis_result = kyc_analyze_with_llm(sanitized_data)

        if analysis_result.get("error"):
            return {
                "error": analysis_result["error"],
                "processing_time": (datetime.now() - start_time).total_seconds(),
            }

        # Step 3: Compliance validation
        validation_result = kyc_compliance_validation(analysis_result)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Step 4: Compile final response
        return {
            "request_data": sanitized_data,
            "initial_analysis": analysis_result,
            "compliance_validation": validation_result,
            "processing_time": processing_time,
            "status": "completed",
        }

    except Exception as e:
        return {
            "error": f"KYC processing failed: {str(e)}",
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "status": "failed",
        }


def extract_kyc_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metrics from KYC processing results.

    Args:
        result: Complete KYC processing results

    Returns:
        Extracted metrics for evaluation
    """
    try:
        # Extract from analysis result
        analysis = result.get("initial_analysis", {}).get("analysis_result", {})
        validation = result.get("compliance_validation", {}).get(
            "validation_result", {}
        )

        return {
            "verdict": validation.get("final_verdict", "REVIEW"),
            "risk_score": analysis.get("risk_assessment", {}).get(
                "overall_risk_score", "Medium"
            ),
            "industry": analysis.get("industry_profile", {}).get(
                "primary_industry", "Unknown"
            ),
            "processing_time": result.get("processing_time", 0.0),
            "tokens_used": (
                result.get("initial_analysis", {}).get("tokens_used", 0)
                + result.get("compliance_validation", {}).get("tokens_used", 0)
            ),
            "confidence_score": validation.get("confidence_score", 0.5),
        }
    except Exception as e:
        return {
            "verdict": "ERROR",
            "risk_score": "High",
            "industry": "Unknown",
            "processing_time": result.get("processing_time", 0.0),
            "tokens_used": 0,
            "confidence_score": 0.0,
            "error": str(e),
        }


@weave.op()
def evaluate_kyc_prediction(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate KYC prediction against expected results.

    Args:
        expected: Expected KYC results
        actual: Actual KYC results from processing

    Returns:
        Evaluation metrics
    """
    metrics = extract_kyc_metrics(actual)

    # Compare results
    verdict_correct = metrics["verdict"] == expected["expected_verdict"]
    risk_correct = metrics["risk_score"] == expected["expected_risk"]
    industry_correct = metrics["industry"] == expected["expected_industry"]

    # Calculate overall score (weighted)
    score_components = [
        (verdict_correct, 0.5),  # Verdict is most important
        (risk_correct, 0.3),  # Risk assessment is critical
        (industry_correct, 0.2),  # Industry classification helps
    ]

    overall_score = sum(correct * weight for correct, weight in score_components)

    return KYCEvaluationResult(
        company_name=expected["business_name"],
        actual_verdict=metrics["verdict"],
        expected_verdict=expected["expected_verdict"],
        actual_risk=metrics["risk_score"],
        expected_risk=expected["expected_risk"],
        actual_industry=metrics["industry"],
        expected_industry=expected["expected_industry"],
        verdict_correct=verdict_correct,
        risk_correct=risk_correct,
        industry_correct=industry_correct,
        overall_score=overall_score,
        processing_time=metrics["processing_time"],
    ).model_dump()


class KYCEvaluation(weave.Evaluation):
    """
    Weave evaluation for KYC processing pipeline.

    Based on the Weave Hello Eval notebook pattern.
    """

    @weave.op()
    def predict(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict function for KYC evaluation."""
        return process_kyc_request_traced(request_data)

    @weave.op()
    def score(
        self, request_data: Dict[str, Any], prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Score function comparing prediction to expected results."""
        return evaluate_kyc_prediction(request_data, prediction)


async def run_kyc_evaluation():
    """
    Run complete KYC evaluation on UK dataset.

    Returns:
        Evaluation results
    """
    print("Starting KYC Evaluation with UK Dataset...")
    print(f"Total companies to evaluate: {len(UK_TEST_COMPANIES)}")

    # Create evaluation instance
    evaluation = KYCEvaluation()

    # Run evaluation
    evaluation_result = await evaluation.evaluate(
        dataset=UK_TEST_COMPANIES, scorers=[evaluate_kyc_prediction]
    )

    print("\n=== KYC Evaluation Results ===")
    print(f"Total evaluations: {len(evaluation_result.rows)}")

    # Calculate aggregate metrics
    scores = [row.scores for row in evaluation_result.rows if row.scores]

    if scores:
        # Overall accuracy metrics
        verdict_accuracy = sum(
            1 for s in scores if s.get("verdict_correct", False)
        ) / len(scores)
        risk_accuracy = sum(1 for s in scores if s.get("risk_correct", False)) / len(
            scores
        )
        industry_accuracy = sum(
            1 for s in scores if s.get("industry_correct", False)
        ) / len(scores)
        avg_overall_score = sum(s.get("overall_score", 0) for s in scores) / len(scores)
        avg_processing_time = sum(s.get("processing_time", 0) for s in scores) / len(
            scores
        )

        print(f"\nAccuracy Metrics:")
        print(f"- Verdict Accuracy: {verdict_accuracy:.2%}")
        print(f"- Risk Assessment Accuracy: {risk_accuracy:.2%}")
        print(f"- Industry Classification Accuracy: {industry_accuracy:.2%}")
        print(f"- Overall Score: {avg_overall_score:.3f}")
        print(f"- Average Processing Time: {avg_processing_time:.2f}s")

        # Detailed results
        print(f"\nDetailed Results:")
        for i, (company, score) in enumerate(zip(UK_TEST_COMPANIES, scores)):
            print(
                f"{i+1}. {company['business_name']}: {score.get('overall_score', 0):.3f}"
            )
            if not score.get("verdict_correct", True):
                print(
                    f"   ❌ Verdict: Expected {company['expected_verdict']}, Got {score.get('actual_verdict', 'N/A')}"
                )
            if not score.get("risk_correct", True):
                print(
                    f"   ❌ Risk: Expected {company['expected_risk']}, Got {score.get('actual_risk', 'N/A')}"
                )

    return evaluation_result


if __name__ == "__main__":
    import asyncio

    # Set OpenAI API key if available
    if not os.getenv("OPENAI_API_KEY"):
        print("Note: Set OPENAI_API_KEY environment variable to avoid prompts")

    # Run evaluation
    asyncio.run(run_kyc_evaluation())
