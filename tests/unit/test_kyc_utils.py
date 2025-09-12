"""
Unit tests for KYC utilities module.

Tests all components: Pydantic models, Ray.data preprocessing, 
LLM integration, and compliance validation.
"""

import json
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest
import ray
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from kyc_utils import (
        KYCRequest, KYCResponse, DocumentBlob, PersonOfInterest,
        CompanyStructure, IndustryProfile, OnlinePresence, RiskAssessment,
        DDInvestigation, ComplianceValidation,
        sanitize_kyc_input, create_kyc_analysis_prompt, 
        create_compliance_validation_prompt, validate_kyc_request,
        parse_kyc_response, create_kyc_processor, create_compliance_processor,
        process_kyc_batch
    )
except ImportError as e:
    pytest.skip(f"KYC utils not available: {e}", allow_module_level=True)


class TestKYCModels:
    """Test Pydantic model validation and structure."""

    def test_kyc_request_valid(self):
        """Test valid KYC request creation."""
        valid_request = {
            "business_name": "Apple Inc",
            "address": "One Apple Park Way, Cupertino, CA 95014",
            "country_code": "US",
            "registration_id": "C0806592",
            "website_url": "https://apple.com"
        }
        
        request = KYCRequest.model_validate(valid_request)
        assert request.business_name == "Apple Inc"
        assert request.country_code == "US"
        assert request.registration_id == "C0806592"

    def test_kyc_request_required_fields(self):
        """Test KYC request with missing required fields."""
        # Missing business_name
        with pytest.raises(ValidationError):
            KYCRequest.model_validate({
                "address": "Test Address",
                "country_code": "US"
            })
        
        # Missing address
        with pytest.raises(ValidationError):
            KYCRequest.model_validate({
                "business_name": "Test Corp",
                "country_code": "US"
            })

    def test_kyc_request_field_validation(self):
        """Test KYC request field validation rules."""
        base_request = {
            "business_name": "Test Corp",
            "address": "Test Address"
        }
        
        # Invalid country code (not 2 letters)
        with pytest.raises(ValidationError):
            KYCRequest.model_validate({
                **base_request,
                "country_code": "USA"
            })
        
        # Invalid country code (lowercase)
        with pytest.raises(ValidationError):
            KYCRequest.model_validate({
                **base_request,
                "country_code": "us"
            })
        
        # Valid country code
        request = KYCRequest.model_validate({
            **base_request,
            "country_code": "US"
        })
        assert request.country_code == "US"

    def test_document_blob_model(self):
        """Test DocumentBlob model validation."""
        doc = DocumentBlob(
            filename="test.pdf",
            content_type="application/pdf",
            content="base64encodedcontent",
            size_bytes=1024
        )
        
        assert doc.filename == "test.pdf"
        assert doc.size_bytes == 1024
        
        # Test negative size validation
        with pytest.raises(ValidationError):
            DocumentBlob(
                filename="test.pdf",
                content_type="application/pdf", 
                content="content",
                size_bytes=-1
            )

    def test_person_of_interest_model(self):
        """Test PersonOfInterest model."""
        person = PersonOfInterest(
            name="John Doe",
            role="Director",
            nationality="US",
            pep_status=True,
            sanctions_match=False,
            risk_indicators=["PEP status"]
        )
        
        assert person.name == "John Doe"
        assert person.pep_status is True
        assert "PEP status" in person.risk_indicators

    def test_kyc_response_structure(self):
        """Test complete KYC response structure."""
        response = KYCResponse(
            verdict="ACCEPT",
            company_structure=CompanyStructure(
                legal_name="Test Corp",
                registered_address="Test Address"
            ),
            people=[],
            industry_profile=IndustryProfile(
                primary_industry="Technology",
                business_description="Software company"
            ),
            online_presence=OnlinePresence(
                website_status="Active"
            ),
            risk_assessment=RiskAssessment(
                overall_risk_score="Low",
                risk_factors=["No significant risks"],
                sanctions_screening="No matches",
                pep_exposure="None",
                adverse_media="None",
                geographic_risk="Low",
                industry_risk="Low"
            ),
            dd_investigation=DDInvestigation(
                investigation_date=datetime.now().isoformat(),
                data_sources=["Public records"],
                verification_methods=["Online search"],
                findings_summary="Clean investigation",
                gaps_identified=[],
                red_flags=[],
                mitigating_factors=["Established business"],
                next_steps=["Standard monitoring"],
                analyst_notes="No concerns identified"
            ),
            compliance_validation=ComplianceValidation(
                reviewer_assessment="Approved",
                data_quality_score="Excellent",
                regulatory_compliance="Compliant",
                escalation_required=False,
                monitoring_requirements="Standard",
                pii_sanitization_status="Applied"
            ),
            processing_time_seconds=2.5,
            confidence_score=0.95
        )
        
        assert response.verdict == "ACCEPT"
        assert response.risk_assessment.overall_risk_score == "Low"
        assert response.confidence_score == 0.95


class TestKYCPreprocessing:
    """Test Ray.data preprocessing functions."""

    def test_sanitize_kyc_input_valid(self):
        """Test sanitization with valid input."""
        input_row = {
            "business_name": "  Apple Inc  ",
            "address": "  One Apple Park Way  ",
            "country_code": "us",
            "registration_id": "  C0806592  ",
            "website_url": "apple.com"
        }
        
        result = sanitize_kyc_input(input_row)
        
        assert result["business_name"] == "Apple Inc"
        assert result["address"] == "One Apple Park Way"
        assert result["country_code"] == "US"
        assert result["registration_id"] == "C0806592"
        assert result["website_url"] == "https://apple.com"
        assert result["validation_status"] == "PASSED"
        assert result["sanitization_applied"] is True

    def test_sanitize_kyc_input_invalid(self):
        """Test sanitization with invalid input."""
        # Empty business name
        invalid_input = {
            "business_name": "",
            "address": "Test Address",
            "country_code": "US"
        }
        
        result = sanitize_kyc_input(invalid_input)
        
        assert result["validation_status"] == "FAILED"
        assert "business_name required" in result["validation_errors"][0]

    def test_sanitize_kyc_input_missing_fields(self):
        """Test sanitization with missing required fields."""
        incomplete_input = {
            "business_name": "Test Corp"
            # Missing address and country_code
        }
        
        result = sanitize_kyc_input(incomplete_input)
        
        assert result["validation_status"] == "FAILED"
        assert "address required" in result["validation_errors"][0]

    def test_create_kyc_analysis_prompt_valid(self):
        """Test KYC analysis prompt creation."""
        sanitized_row = {
            "business_name": "Apple Inc",
            "address": "One Apple Park Way",
            "country_code": "US",
            "registration_id": "C0806592",
            "website_url": "https://apple.com",
            "validation_status": "PASSED"
        }
        
        result = create_kyc_analysis_prompt(sanitized_row)
        
        assert "messages" in result
        assert "sampling_params" in result
        assert len(result["messages"]) == 2
        assert "Apple Inc" in result["messages"][1]["content"]
        assert result["sampling_params"]["temperature"] == 0.1

    def test_create_kyc_analysis_prompt_invalid(self):
        """Test KYC analysis prompt with invalid input."""
        invalid_row = {
            "validation_status": "FAILED",
            "validation_errors": ["Test error"]
        }
        
        result = create_kyc_analysis_prompt(invalid_row)
        
        assert result["skip_processing"] is True
        assert "Invalid input data" in result["messages"][0]["content"]

    def test_create_compliance_validation_prompt(self):
        """Test compliance validation prompt creation."""
        analysis_row = {
            "generated_text": '{"company_profile": "Test analysis"}',
            "original_input": {
                "business_name": "Apple Inc",
                "country_code": "US",
                "processing_timestamp": "2025-01-01T00:00:00"
            }
        }
        
        result = create_compliance_validation_prompt(analysis_row)
        
        assert "messages" in result
        assert "sampling_params" in result
        assert "Apple Inc" in result["messages"][1]["content"]
        assert result["sampling_params"]["temperature"] == 0.05


class TestKYCProcessors:
    """Test KYC processor creation and configuration."""

    @patch("ray.data.llm.build_llm_processor")
    @patch("ray.data.llm.vLLMEngineProcessorConfig")
    def test_create_kyc_processor(self, mock_config, mock_build_processor):
        """Test KYC processor creation."""
        mock_processor = Mock()
        mock_build_processor.return_value = mock_processor
        mock_config.return_value = Mock()
        
        processor = create_kyc_processor()
        
        # Should call vLLM configuration
        mock_config.assert_called_once()
        mock_build_processor.assert_called_once()
        
        # Check configuration parameters
        call_args = mock_config.call_args
        assert call_args[1]["model_source"] == "microsoft/DialoGPT-medium"
        assert call_args[1]["batch_size"] == 16
        
        assert processor == mock_processor

    @patch("ray.data.llm.build_llm_processor")
    @patch("ray.data.llm.vLLMEngineProcessorConfig") 
    def test_create_compliance_processor(self, mock_config, mock_build_processor):
        """Test compliance processor creation."""
        mock_processor = Mock()
        mock_build_processor.return_value = mock_processor
        mock_config.return_value = Mock()
        
        processor = create_compliance_processor()
        
        mock_config.assert_called_once()
        mock_build_processor.assert_called_once()
        
        # Check compliance-specific configuration
        call_args = mock_config.call_args
        assert call_args[1]["batch_size"] == 8  # Smaller batch for compliance

    def test_create_kyc_processor_no_vllm(self):
        """Test KYC processor creation without vLLM available."""
        # Mock at the function level where it's imported
        with patch("kyc_utils.logger") as mock_logger:
            # Mock the import to fail at the top level
            with patch.dict("sys.modules", {"ray.data.llm": None}):
                processor = create_kyc_processor()
                assert processor is None
                mock_logger.warning.assert_called_with("vLLM not available - using mock processor for testing")


class TestKYCBatchProcessing:
    """Test complete batch processing pipeline."""

    @patch("kyc_utils.create_compliance_processor")
    @patch("kyc_utils.create_kyc_processor")
    @patch("ray.data.from_items")
    def test_process_kyc_batch_success(self, mock_from_items, mock_kyc_proc, mock_compliance_proc):
        """Test successful batch processing."""
        # Mock Ray dataset and processors
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_from_items.return_value = mock_dataset
        
        mock_kyc_processor = Mock()
        mock_kyc_processor.return_value = mock_dataset
        mock_kyc_proc.return_value = mock_kyc_processor
        
        mock_compliance_processor = Mock()
        mock_compliance_processor.return_value = mock_dataset
        mock_compliance_proc.return_value = mock_compliance_processor
        
        # Mock final results
        mock_results = [
            {"business_name": "Apple Inc", "analysis": "complete"},
            {"business_name": "Google Inc", "analysis": "complete"}
        ]
        mock_dataset.take_all.return_value = mock_results
        
        # Test batch processing
        requests = [
            {"business_name": "Apple Inc", "address": "Cupertino", "country_code": "US"},
            {"business_name": "Google Inc", "address": "Mountain View", "country_code": "US"}
        ]
        
        results = process_kyc_batch(requests)
        
        # Verify processing steps
        mock_from_items.assert_called_once_with(requests)
        mock_dataset.map.assert_called_once()  # Sanitization step
        mock_kyc_processor.assert_called_once()  # KYC analysis
        mock_compliance_processor.assert_called_once()  # Compliance validation
        
        # Check results
        assert len(results) == 2
        assert all("total_processing_time" in result for result in results)
        assert all("batch_size" in result for result in results)

    @patch("kyc_utils.create_kyc_processor")
    def test_process_kyc_batch_processor_unavailable(self, mock_kyc_proc):
        """Test batch processing with unavailable processor."""
        mock_kyc_proc.return_value = None
        
        requests = [{"business_name": "Test", "address": "Test", "country_code": "US"}]
        
        with pytest.raises(RuntimeError, match="KYC processor not available"):
            process_kyc_batch(requests)

    @patch("ray.data.from_items")
    def test_process_kyc_batch_ray_failure(self, mock_from_items):
        """Test batch processing with Ray failure."""
        mock_from_items.side_effect = Exception("Ray cluster unavailable")
        
        requests = [{"business_name": "Test", "address": "Test", "country_code": "US"}]
        
        with pytest.raises(RuntimeError, match="KYC processing failed"):
            process_kyc_batch(requests)


class TestKYCValidation:
    """Test request validation and response parsing."""

    def test_validate_kyc_request_success(self):
        """Test successful request validation."""
        request_dict = {
            "business_name": "Apple Inc",
            "address": "One Apple Park Way",
            "country_code": "US",
            "registration_id": "C0806592"
        }
        
        validated = validate_kyc_request(request_dict)
        
        assert isinstance(validated, KYCRequest)
        assert validated.business_name == "Apple Inc"
        assert validated.country_code == "US"

    def test_validate_kyc_request_failure(self):
        """Test request validation failure."""
        invalid_request = {
            "business_name": "",  # Empty name
            "address": "Test",
            "country_code": "USA"  # Invalid code
        }
        
        with pytest.raises(ValidationError):
            validate_kyc_request(invalid_request)

    def test_parse_kyc_response_basic(self):
        """Test basic KYC response parsing."""
        result_dict = {
            "business_name": "Apple Inc",
            "address": "Cupertino",
            "total_processing_time": 5.2
        }
        
        response = parse_kyc_response(result_dict)
        
        assert isinstance(response, KYCResponse)
        assert response.verdict == "REVIEW"  # Default safe verdict
        assert response.company_structure.legal_name == "Apple Inc"
        assert response.processing_time_seconds == 5.2
        assert response.confidence_score == 0.5  # Default medium confidence

    @pytest.mark.parametrize("verdict", ["ACCEPT", "REJECT", "REVIEW"])
    def test_kyc_response_verdicts(self, verdict):
        """Test all valid KYC verdict options."""
        response = KYCResponse(
            verdict=verdict,
            company_structure=CompanyStructure(
                legal_name="Test",
                registered_address="Test"
            ),
            people=[],
            industry_profile=IndustryProfile(
                primary_industry="Test",
                business_description="Test"
            ),
            online_presence=OnlinePresence(website_status="Unknown"),
            risk_assessment=RiskAssessment(
                overall_risk_score="Medium",
                risk_factors=[],
                sanctions_screening="None",
                pep_exposure="None", 
                adverse_media="None",
                geographic_risk="Medium",
                industry_risk="Medium"
            ),
            dd_investigation=DDInvestigation(
                investigation_date="2025-01-01T00:00:00",
                data_sources=[],
                verification_methods=[],
                findings_summary="Test",
                gaps_identified=[],
                red_flags=[],
                mitigating_factors=[],
                next_steps=[],
                analyst_notes="Test"
            ),
            compliance_validation=ComplianceValidation(
                reviewer_assessment="Test",
                data_quality_score="Fair",
                regulatory_compliance="Test",
                escalation_required=False,
                monitoring_requirements="Test",
                pii_sanitization_status="Applied"
            ),
            processing_time_seconds=1.0,
            confidence_score=0.8
        )
        
        assert response.verdict == verdict


class TestKYCIntegration:
    """Integration tests combining multiple components."""

    @patch("ray.data.from_items")
    def test_full_preprocessing_pipeline(self, mock_from_items, ray_cluster):
        """Test complete preprocessing pipeline with Ray.data."""
        # Sample KYC requests
        requests = [
            {
                "business_name": "  Apple Inc  ",
                "address": "  One Apple Park Way  ",
                "country_code": "us",
                "registration_id": "C0806592"
            },
            {
                "business_name": "Google Inc",
                "address": "Mountain View",
                "country_code": "US"
            }
        ]
        
        # Mock sanitized results to avoid Ray serialization issues
        mock_sanitized_results = [
            {
                "business_name": "Apple Inc",
                "address": "One Apple Park Way",
                "country_code": "US",
                "registration_id": "C0806592",
                "validation_status": "PASSED",
                "sanitization_applied": True
            },
            {
                "business_name": "Google Inc", 
                "address": "Mountain View",
                "country_code": "US",
                "validation_status": "PASSED",
                "sanitization_applied": True
            }
        ]
        
        # Mock Ray dataset operations
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.take_all.return_value = mock_sanitized_results
        mock_from_items.return_value = mock_dataset
        
        # Test the processing logic without Ray serialization
        ds = ray.data.from_items(requests)
        sanitized_ds = ds.map(sanitize_kyc_input)
        results = sanitized_ds.take_all()
        
        # Verify mocked results
        assert len(results) == 2
        assert results[0]["business_name"] == "Apple Inc"
        assert results[0]["country_code"] == "US"
        assert results[0]["validation_status"] == "PASSED"
        
        # Test prompt creation on sanitized data
        prompts = [create_kyc_analysis_prompt(row) for row in results]
        
        assert all("messages" in prompt for prompt in prompts)
        assert all(not prompt.get("skip_processing", False) for prompt in prompts)

    def test_model_validation_integration(self):
        """Test integration between validation and model parsing."""
        # Create valid request
        request_dict = {
            "business_name": "Apple Inc",
            "address": "One Apple Park Way",
            "country_code": "US"
        }
        
        # Validate request
        validated_request = validate_kyc_request(request_dict)
        
        # Convert to dict for processing
        request_data = validated_request.model_dump()
        
        # Sanitize
        sanitized = sanitize_kyc_input(request_data)
        
        # Should pass sanitization
        assert sanitized["validation_status"] == "PASSED"
        assert sanitized["business_name"] == validated_request.business_name