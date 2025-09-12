"""
Unit tests for KYC utilities with Weave tracing integration.

Tests the enhanced KYC processing pipeline with OpenAI LLM integration,
Weave tracing, and evaluation capabilities.
"""

import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from kyc_weave import (
        UK_TEST_COMPANIES,
        KYCEvaluationResult,
        kyc_analyze_with_llm,
        kyc_compliance_validation,
        process_kyc_request_traced,
        extract_kyc_metrics,
        evaluate_kyc_prediction,
        KYCEvaluation,
    )
except ImportError as e:
    pytest.skip(f"KYC Weave utilities not available: {e}", allow_module_level=True)


class TestUKDataset:
    """Test UK company dataset for KYC evaluation."""

    def test_uk_dataset_structure(self):
        """Test UK dataset has required structure."""
        assert len(UK_TEST_COMPANIES) == 10

        required_fields = [
            "business_name",
            "address",
            "country_code",
            "registration_id",
            "website_url",
            "expected_verdict",
            "expected_risk",
            "expected_industry",
        ]

        for company in UK_TEST_COMPANIES:
            for field in required_fields:
                assert (
                    field in company
                ), f"Missing {field} in {company.get('business_name', 'Unknown')}"

            # Validate field values
            assert company["country_code"] == "GB"
            assert company["expected_verdict"] in ["ACCEPT", "REJECT", "REVIEW"]
            assert company["expected_risk"] in ["Low", "Medium", "High"]
            assert len(company["business_name"]) > 0
            assert len(company["address"]) > 0

    def test_uk_companies_variety(self):
        """Test UK dataset covers various industries and risk levels."""
        industries = [company["expected_industry"] for company in UK_TEST_COMPANIES]
        verdicts = [company["expected_verdict"] for company in UK_TEST_COMPANIES]
        risks = [company["expected_risk"] for company in UK_TEST_COMPANIES]

        # Should have variety in classifications
        assert len(set(industries)) >= 5  # At least 5 different industries
        assert "ACCEPT" in verdicts  # Some should be acceptable
        assert len(set(risks)) >= 2  # At least 2 different risk levels

    def test_uk_company_examples(self):
        """Test specific UK company examples."""
        company_names = [company["business_name"] for company in UK_TEST_COMPANIES]

        # Should include major UK companies
        assert "Tesco PLC" in company_names
        assert "HSBC Holdings plc" in company_names
        assert "BP p.l.c." in company_names

        # Check Tesco specifically
        tesco = next(c for c in UK_TEST_COMPANIES if c["business_name"] == "Tesco PLC")
        assert tesco["registration_id"] == "00445790"
        assert tesco["expected_verdict"] == "ACCEPT"
        assert tesco["expected_risk"] == "Low"


class TestKYCEvaluationModels:
    """Test evaluation data models."""

    def test_kyc_evaluation_result_model(self):
        """Test KYCEvaluationResult model validation."""
        result = KYCEvaluationResult(
            company_name="Test Corp",
            actual_verdict="ACCEPT",
            expected_verdict="ACCEPT",
            actual_risk="Low",
            expected_risk="Low",
            actual_industry="Technology",
            expected_industry="Technology",
            verdict_correct=True,
            risk_correct=True,
            industry_correct=True,
            overall_score=1.0,
            processing_time=2.5,
        )

        assert result.company_name == "Test Corp"
        assert result.verdict_correct is True
        assert result.overall_score == 1.0
        assert result.processing_time == 2.5

    def test_kyc_evaluation_result_scoring(self):
        """Test evaluation result score validation."""
        # Test score boundaries
        result = KYCEvaluationResult(
            company_name="Test",
            actual_verdict="ACCEPT",
            expected_verdict="REJECT",
            actual_risk="Low",
            expected_risk="High",
            actual_industry="Tech",
            expected_industry="Finance",
            verdict_correct=False,
            risk_correct=False,
            industry_correct=False,
            overall_score=0.0,  # Minimum score
            processing_time=1.0,
        )

        assert result.overall_score == 0.0

        # Test invalid scores raise validation errors
        with pytest.raises(ValueError):
            KYCEvaluationResult(
                company_name="Test",
                actual_verdict="ACCEPT",
                expected_verdict="ACCEPT",
                actual_risk="Low",
                expected_risk="Low",
                actual_industry="Tech",
                expected_industry="Tech",
                verdict_correct=True,
                risk_correct=True,
                industry_correct=True,
                overall_score=1.5,  # Invalid > 1.0
                processing_time=1.0,
            )


class TestKYCWeaveOps:
    """Test Weave-decorated operations."""

    @patch("openai.OpenAI")
    @patch("os.getenv")
    def test_kyc_analyze_with_llm_success(self, mock_getenv, mock_openai):
        """Test successful LLM analysis with mocked OpenAI."""
        mock_getenv.return_value = "test-api-key"

        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "risk_assessment": {"overall_risk_score": "Low"},
                "industry_profile": {"primary_industry": "Technology"},
            }
        )
        mock_response.usage.total_tokens = 1500
        mock_client.chat.completions.create.return_value = mock_response

        # Test input data
        input_data = {
            "business_name": "Test Corp",
            "address": "Test Address",
            "country_code": "US",
            "validation_status": "PASSED",
        }

        result = kyc_analyze_with_llm(input_data)

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"
        assert call_args[1]["response_format"] == {"type": "json_object"}

        # Verify result
        assert "analysis_result" in result
        assert result["model"] == "gpt-4o-mini"
        assert result["tokens_used"] == 1500
        assert result["processing_stage"] == "initial_analysis"

    @patch("openai.OpenAI")
    @patch("os.getenv")
    def test_kyc_analyze_with_llm_invalid_input(self, mock_getenv, mock_openai):
        """Test LLM analysis with invalid input."""
        mock_getenv.return_value = "test-api-key"

        invalid_input = {"business_name": "", "validation_status": "FAILED"}

        result = kyc_analyze_with_llm(invalid_input)

        assert result["skip_processing"] is True
        assert "error" in result

    @patch("openai.OpenAI")
    @patch("os.getenv")
    def test_kyc_compliance_validation_success(self, mock_getenv, mock_openai):
        """Test successful compliance validation."""
        mock_getenv.return_value = "test-api-key"

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "final_verdict": "ACCEPT",
                "confidence_score": 0.9,
                "dd_investigation": {"investigation_date": "2025-01-01"},
            }
        )
        mock_response.usage.total_tokens = 800
        mock_client.chat.completions.create.return_value = mock_response

        # Test analysis input
        analysis_result = {
            "analysis_result": {"risk_assessment": {"overall_risk_score": "Low"}},
            "original_input": {"business_name": "Test Corp"},
        }

        result = kyc_compliance_validation(analysis_result)

        # Verify result
        assert "validation_result" in result
        assert result["model"] == "gpt-4o-mini"
        assert result["tokens_used"] == 800
        assert result["processing_stage"] == "compliance_validation"

    @patch("kyc_weave.kyc_compliance_validation")
    @patch("kyc_weave.kyc_analyze_with_llm")
    @patch("kyc_weave.validate_kyc_request")
    @patch("kyc_weave.sanitize_kyc_input")
    def test_process_kyc_request_traced(
        self, mock_sanitize, mock_validate, mock_analyze, mock_compliance
    ):
        """Test complete traced KYC processing."""
        # Mock all steps
        mock_validate.return_value = Mock()
        mock_validate.return_value.model_dump.return_value = {
            "business_name": "Test Corp",
            "country_code": "US",
        }

        mock_sanitize.return_value = {
            "validation_status": "PASSED",
            "business_name": "Test Corp",
        }

        mock_analyze.return_value = {
            "analysis_result": {"risk_assessment": {"overall_risk_score": "Low"}},
            "tokens_used": 1000,
        }

        mock_compliance.return_value = {
            "validation_result": {"final_verdict": "ACCEPT"},
            "tokens_used": 500,
        }

        # Test request
        request_data = {
            "business_name": "Test Corp",
            "address": "Test Address",
            "country_code": "US",
        }

        result = process_kyc_request_traced(request_data)

        # Verify all steps called
        mock_validate.assert_called_once_with(request_data)
        mock_sanitize.assert_called_once()
        mock_analyze.assert_called_once()
        mock_compliance.assert_called_once()

        # Verify result structure
        assert result["status"] == "completed"
        assert "processing_time" in result
        assert "request_data" in result
        assert "initial_analysis" in result
        assert "compliance_validation" in result


class TestKYCEvaluationLogic:
    """Test KYC evaluation and scoring logic."""

    def test_extract_kyc_metrics_complete(self):
        """Test metric extraction from complete results."""
        complete_result = {
            "initial_analysis": {
                "analysis_result": {
                    "risk_assessment": {"overall_risk_score": "Low"},
                    "industry_profile": {"primary_industry": "Technology"},
                },
                "tokens_used": 1000,
            },
            "compliance_validation": {
                "validation_result": {
                    "final_verdict": "ACCEPT",
                    "confidence_score": 0.9,
                },
                "tokens_used": 500,
            },
            "processing_time": 5.2,
        }

        metrics = extract_kyc_metrics(complete_result)

        assert metrics["verdict"] == "ACCEPT"
        assert metrics["risk_score"] == "Low"
        assert metrics["industry"] == "Technology"
        assert metrics["processing_time"] == 5.2
        assert metrics["tokens_used"] == 1500  # Sum of both
        assert metrics["confidence_score"] == 0.9

    def test_extract_kyc_metrics_incomplete(self):
        """Test metric extraction from incomplete results."""
        incomplete_result = {"processing_time": 2.0, "error": "Analysis failed"}

        metrics = extract_kyc_metrics(incomplete_result)

        # Should provide defaults
        assert metrics["verdict"] == "REVIEW"
        assert metrics["risk_score"] == "Medium"
        assert metrics["industry"] == "Unknown"
        assert metrics["processing_time"] == 2.0
        assert metrics["tokens_used"] == 0
        assert metrics["confidence_score"] == 0.5

    def test_evaluate_kyc_prediction_perfect(self):
        """Test evaluation with perfect prediction."""
        expected = {
            "business_name": "Test Corp",
            "expected_verdict": "ACCEPT",
            "expected_risk": "Low",
            "expected_industry": "Technology",
        }

        actual = {
            "initial_analysis": {
                "analysis_result": {
                    "risk_assessment": {"overall_risk_score": "Low"},
                    "industry_profile": {"primary_industry": "Technology"},
                }
            },
            "compliance_validation": {"validation_result": {"final_verdict": "ACCEPT"}},
            "processing_time": 3.0,
        }

        evaluation = evaluate_kyc_prediction(expected, actual)

        assert evaluation["company_name"] == "Test Corp"
        assert evaluation["verdict_correct"] is True
        assert evaluation["risk_correct"] is True
        assert evaluation["industry_correct"] is True
        assert evaluation["overall_score"] == 1.0  # Perfect score

    def test_evaluate_kyc_prediction_partial(self):
        """Test evaluation with partial correct prediction."""
        expected = {
            "business_name": "Test Corp",
            "expected_verdict": "ACCEPT",
            "expected_risk": "Low",
            "expected_industry": "Technology",
        }

        actual = {
            "initial_analysis": {
                "analysis_result": {
                    "risk_assessment": {"overall_risk_score": "Medium"},  # Wrong
                    "industry_profile": {"primary_industry": "Technology"},  # Correct
                }
            },
            "compliance_validation": {
                "validation_result": {"final_verdict": "ACCEPT"}  # Correct
            },
            "processing_time": 3.0,
        }

        evaluation = evaluate_kyc_prediction(expected, actual)

        assert evaluation["verdict_correct"] is True  # 0.5 weight
        assert evaluation["risk_correct"] is False  # 0.3 weight
        assert evaluation["industry_correct"] is True  # 0.2 weight
        # Score = 0.5*1 + 0.3*0 + 0.2*1 = 0.7
        assert evaluation["overall_score"] == 0.7

    def test_evaluate_kyc_prediction_failed(self):
        """Test evaluation with completely wrong prediction."""
        expected = {
            "business_name": "Test Corp",
            "expected_verdict": "ACCEPT",
            "expected_risk": "Low",
            "expected_industry": "Technology",
        }

        actual = {"error": "Processing failed", "processing_time": 1.0}

        evaluation = evaluate_kyc_prediction(expected, actual)

        # Should get defaults which are wrong
        assert evaluation["verdict_correct"] is False  # REVIEW != ACCEPT
        assert evaluation["risk_correct"] is False  # Medium != Low
        assert evaluation["industry_correct"] is False  # Unknown != Technology
        assert evaluation["overall_score"] == 0.0  # All wrong


class TestKYCEvaluationClass:
    """Test KYC evaluation class."""

    def test_kyc_evaluation_instantiation(self):
        """Test KYCEvaluation class can be instantiated."""
        evaluation = KYCEvaluation()

        # Should have required methods
        assert hasattr(evaluation, "predict")
        assert hasattr(evaluation, "score")

    @patch("kyc_weave.process_kyc_request_traced")
    def test_kyc_evaluation_predict(self, mock_process):
        """Test evaluation predict method."""
        mock_process.return_value = {"status": "completed", "processing_time": 2.0}

        evaluation = KYCEvaluation()
        request_data = {"business_name": "Test Corp"}

        result = evaluation.predict(request_data)

        mock_process.assert_called_once_with(request_data)
        assert result["status"] == "completed"

    @patch("kyc_weave.evaluate_kyc_prediction")
    def test_kyc_evaluation_score(self, mock_evaluate):
        """Test evaluation score method."""
        mock_evaluate.return_value = {"overall_score": 0.8, "verdict_correct": True}

        evaluation = KYCEvaluation()
        request_data = {"business_name": "Test Corp"}
        prediction = {"status": "completed"}

        result = evaluation.score(request_data, prediction)

        mock_evaluate.assert_called_once_with(request_data, prediction)
        assert result["overall_score"] == 0.8


@pytest.mark.asyncio
class TestKYCEvaluationAsync:
    """Test async evaluation functionality."""

    @patch("kyc_weave.KYCEvaluation")
    async def test_run_kyc_evaluation_structure(self, mock_evaluation_class):
        """Test run_kyc_evaluation basic structure."""
        # This would test the actual async evaluation
        # For now, just test that it can be imported and structured correctly
        from kyc_weave import run_kyc_evaluation

        # Mock the evaluation
        mock_evaluation = Mock()
        mock_result = Mock()
        mock_result.rows = []
        mock_evaluation.evaluate.return_value = mock_result
        mock_evaluation_class.return_value = mock_evaluation

        # This would run the full evaluation but is mocked
        # result = await run_kyc_evaluation()

        # For now, just verify the function exists and is async
        import inspect

        assert inspect.iscoroutinefunction(run_kyc_evaluation)


class TestKYCIntegrationWithWeave:
    """Integration tests for KYC with Weave tracing."""

    @patch("weave.init")
    def test_weave_initialization(self, mock_weave_init):
        """Test that Weave is properly initialized."""
        # Import should trigger weave.init("KYC")
        import kyc_weave

        # Verify Weave was initialized (would be called during import)
        # This test verifies the import worked without errors

    def test_uk_dataset_integration(self):
        """Test UK dataset integrates properly with evaluation."""
        # Get a sample company
        sample_company = UK_TEST_COMPANIES[0]

        # Should be able to extract evaluation metrics
        mock_result = {
            "initial_analysis": {
                "analysis_result": {
                    "industry_profile": {
                        "primary_industry": sample_company["expected_industry"]
                    }
                }
            },
            "compliance_validation": {
                "validation_result": {
                    "final_verdict": sample_company["expected_verdict"]
                }
            },
            "processing_time": 2.0,
        }

        evaluation = evaluate_kyc_prediction(sample_company, mock_result)

        # Should be valid evaluation result
        assert "overall_score" in evaluation
        assert "company_name" in evaluation
        assert evaluation["company_name"] == sample_company["business_name"]
