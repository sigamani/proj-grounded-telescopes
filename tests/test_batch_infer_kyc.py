"""
Comprehensive pytest test suite for batch inference with KYC sanitization integration.
"""

import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import functions to test
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.kyc_utils import sanitize_kyc_input
except ImportError:
    # Fallback for testing without Ray dependencies
    from test_kyc_sanitization import sanitize_kyc_input


class TestKYCSanitization:
    """Test cases for KYC input sanitization."""

    def test_valid_input_sanitization(self):
        """Test sanitization with valid input containing formatting issues."""
        input_data = {
            "business_name": "   Acme Financial Services Ltd   ",
            "address": "  123 Main Street, London, UK  ",
            "country_code": "gb",
            "registration_id": "  12345678  ",
            "website_url": "acmefinance.com",
        }

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "PASSED"
        assert result["business_name"] == "Acme Financial Services Ltd"
        assert result["address"] == "123 Main Street, London, UK"
        assert result["country_code"] == "GB"
        assert result["registration_id"] == "12345678"
        assert result["website_url"] == "https://acmefinance.com"
        assert result["sanitization_applied"] is True
        assert result["pii_scrubbing_status"] == "APPLIED"
        assert "processing_timestamp" in result

    def test_missing_business_name_validation(self):
        """Test validation failure when business name is missing."""
        input_data = {
            "business_name": "",
            "address": "Valid address",
            "country_code": "US",
        }

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "FAILED"
        assert "business_name required and non-empty" in result["validation_errors"]

    def test_missing_address_validation(self):
        """Test validation failure when address is missing."""
        input_data = {
            "business_name": "Valid Company",
            "address": "   ",
            "country_code": "US",
        }

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "FAILED"
        assert "address required and non-empty" in result["validation_errors"]

    def test_invalid_country_code_validation(self):
        """Test validation failure with invalid country code."""
        test_cases = [
            {"country_code": "INVALID"},  # Too long
            {"country_code": "U"},  # Too short
            {"country_code": "12"},  # Numbers
            {"country_code": "U$"},  # Special chars
        ]

        for case in test_cases:
            input_data = {
                "business_name": "Valid Company",
                "address": "Valid address",
                **case,
            }

            result = sanitize_kyc_input(input_data)

            assert result["validation_status"] == "FAILED"
            assert (
                "country_code must be valid ISO2 format" in result["validation_errors"]
            )

    def test_url_sanitization(self):
        """Test URL sanitization adds https:// protocol."""
        input_data = {
            "business_name": "Tech Corp",
            "address": "456 Tech St",
            "country_code": "US",
            "website_url": "techcorp.com",
        }

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "PASSED"
        assert result["website_url"] == "https://techcorp.com"

    def test_url_with_existing_protocol(self):
        """Test URL sanitization preserves existing protocol."""
        input_data = {
            "business_name": "Tech Corp",
            "address": "456 Tech St",
            "country_code": "US",
            "website_url": "https://techcorp.com",
        }

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "PASSED"
        assert result["website_url"] == "https://techcorp.com"

    def test_optional_fields_handling(self):
        """Test handling of optional fields."""
        input_data = {
            "business_name": "Minimal Company",
            "address": "123 Basic St",
            "country_code": "US",
        }

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "PASSED"
        assert result["registration_id"] == ""
        assert result["website_url"] == ""

    def test_special_characters_preservation(self):
        """Test that legitimate special characters are preserved."""
        input_data = {
            "business_name": "Company & Co., Ltd.",
            "address": "789 Street, Apt #5, Suite A-1",
            "country_code": "FR",
            "registration_id": "FR-123-456-789",
            "website_url": "https://company-co.fr/path?param=value",
        }

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "PASSED"
        assert result["business_name"] == "Company & Co., Ltd."
        assert result["address"] == "789 Street, Apt #5, Suite A-1"
        assert result["registration_id"] == "FR-123-456-789"
        assert result["website_url"] == "https://company-co.fr/path?param=value"

    def test_exception_handling(self):
        """Test exception handling in sanitization function."""
        # Simulate an error by passing None which will cause an exception
        with patch("test_kyc_sanitization.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Simulated error")

            result = sanitize_kyc_input({"business_name": "Test"})

            assert result["validation_status"] == "ERROR"
            assert "Sanitization error" in result["validation_errors"][0]


class TestBatchIntegration:
    """Test cases for batch integration with KYC sanitization."""

    def test_batch_processing_mock(self):
        """Test batch processing with multiple KYC inputs."""
        batch_data = [
            {
                "business_name": "   Acme Financial Services Ltd   ",
                "address": "  123 Main Street, London, UK  ",
                "country_code": "gb",
                "website_url": "acmefinance.com",
            },
            {
                "business_name": "Tech Startup Inc",
                "address": "456 Innovation Drive",
                "country_code": "us",
                "website_url": "techstartup.com",
            },
            {
                "business_name": "",  # Invalid case
                "address": "Some address",
                "country_code": "CA",
            },
        ]

        results = []
        for data in batch_data:
            result = sanitize_kyc_input(data)
            results.append(result)

        # Test results
        assert len(results) == 3

        # First item should pass
        assert results[0]["validation_status"] == "PASSED"
        assert results[0]["business_name"] == "Acme Financial Services Ltd"
        assert results[0]["country_code"] == "GB"
        assert results[0]["website_url"] == "https://acmefinance.com"

        # Second item should pass
        assert results[1]["validation_status"] == "PASSED"
        assert results[1]["business_name"] == "Tech Startup Inc"
        assert results[1]["country_code"] == "US"

        # Third item should fail (missing business name)
        assert results[2]["validation_status"] == "FAILED"
        assert "business_name required and non-empty" in results[2]["validation_errors"]

        # Count validation results
        passed_count = sum(1 for r in results if r["validation_status"] == "PASSED")
        failed_count = sum(1 for r in results if r["validation_status"] == "FAILED")

        assert passed_count == 2
        assert failed_count == 1


class TestBatchInferenceIntegration:
    """Test cases for batch inference integration."""

    @patch("src.batch_infer.ray")
    def test_batch_processor_creation_mock(self, mock_ray):
        """Test batch processor creation with mocked dependencies."""
        # Mock Ray initialization
        mock_ray.is_initialized.return_value = False
        mock_ray.init = Mock()

        # Mock vLLM components
        mock_config = Mock()
        mock_processor = Mock()

        with patch("src.batch_infer.build_llm_processor", return_value=mock_processor):
            with patch(
                "src.batch_infer.vLLMEngineProcessorConfig", return_value=mock_config
            ):
                from src.batch_infer import create_batch_processor

                result = create_batch_processor()

                assert result == mock_processor
                mock_ray.init.assert_called_once()

    @patch("src.batch_infer.ray")
    def test_batch_inference_with_kyc_data_mock(self, mock_ray):
        """Test batch inference run with KYC sanitization."""
        # Mock Ray components
        mock_ray.is_initialized.return_value = True
        mock_ds = Mock()
        mock_processor = Mock()

        mock_ray.data.from_items.return_value = mock_ds
        mock_ds.write_json = Mock()

        with patch(
            "src.batch_infer.create_batch_processor", return_value=mock_processor
        ):
            with patch("src.batch_infer.sanitize_kyc_input") as mock_sanitize:
                # Setup mock sanitization result
                mock_sanitize.return_value = {
                    "business_name": "Acme Financial Services Ltd",
                    "address": "123 Main Street, London, UK",
                    "country_code": "GB",
                    "validation_status": "PASSED",
                    "sanitization_applied": True,
                }

                from src.batch_infer import run_batch_inference

                # This should not raise an exception
                run_batch_inference()

                # Verify sanitization was called
                mock_sanitize.assert_called_once()

                # Verify Ray dataset creation
                mock_ray.data.from_items.assert_called_once()

                # Verify processor was called
                mock_processor.assert_called_once_with(mock_ds)

    def test_integration_data_flow(self):
        """Test data flow from input through sanitization to batch processing."""
        # Test the complete data flow without heavy dependencies
        original_data = {
            "business_name": "   Acme Corp   ",
            "address": "  123 Main St  ",
            "country_code": "us",
            "website_url": "acme.com",
            "prompt": "Analyze KYC requirements.",
        }

        # Step 1: Sanitization
        sanitized_data = sanitize_kyc_input(original_data)

        # Verify sanitization worked
        assert sanitized_data["validation_status"] == "PASSED"
        assert sanitized_data["business_name"] == "Acme Corp"
        assert sanitized_data["country_code"] == "US"
        assert sanitized_data["website_url"] == "https://acme.com"

        # Step 2: Mock batch processing
        batch_items = [sanitized_data]

        # This would be passed to Ray.data.from_items() in real implementation
        assert len(batch_items) == 1
        assert batch_items[0]["validation_status"] == "PASSED"

        # Step 3: Verify processing metadata is present
        assert "processing_timestamp" in sanitized_data
        assert "sanitization_applied" in sanitized_data
        assert "pii_scrubbing_status" in sanitized_data


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_input(self):
        """Test handling of empty input."""
        result = sanitize_kyc_input({})

        assert result["validation_status"] == "FAILED"
        assert "business_name required and non-empty" in result["validation_errors"]

    def test_none_values(self):
        """Test handling of None values."""
        input_data = {"business_name": None, "address": None, "country_code": None}

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "FAILED"
        # Should fail on business_name first
        assert "business_name required and non-empty" in result["validation_errors"]

    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        input_data = {
            "business_name": "Compañía Española S.A.",
            "address": "Calle José María, 123, Madrid",
            "country_code": "ES",
            "website_url": "empresa-española.com",
        }

        result = sanitize_kyc_input(input_data)

        assert result["validation_status"] == "PASSED"
        assert result["business_name"] == "Compañía Española S.A."
        assert result["address"] == "Calle José María, 123, Madrid"
        assert result["website_url"] == "https://empresa-española.com"


if __name__ == "__main__":
    # Run tests directly without pytest since it's not available
    import unittest

    class TestRunner(unittest.TestCase):
        def setUp(self):
            self.kyc_test = TestKYCSanitization()
            self.batch_test = TestBatchIntegration()
            self.edge_test = TestEdgeCases()

        def test_all_kyc_sanitization(self):
            """Run all KYC sanitization tests."""
            try:
                self.kyc_test.test_valid_input_sanitization()
                print("✅ test_valid_input_sanitization PASSED")
            except Exception as e:
                print(f"❌ test_valid_input_sanitization FAILED: {e}")

            try:
                self.kyc_test.test_missing_business_name_validation()
                print("✅ test_missing_business_name_validation PASSED")
            except Exception as e:
                print(f"❌ test_missing_business_name_validation FAILED: {e}")

            try:
                self.kyc_test.test_missing_address_validation()
                print("✅ test_missing_address_validation PASSED")
            except Exception as e:
                print(f"❌ test_missing_address_validation FAILED: {e}")

            try:
                self.kyc_test.test_invalid_country_code_validation()
                print("✅ test_invalid_country_code_validation PASSED")
            except Exception as e:
                print(f"❌ test_invalid_country_code_validation FAILED: {e}")

            try:
                self.kyc_test.test_url_sanitization()
                print("✅ test_url_sanitization PASSED")
            except Exception as e:
                print(f"❌ test_url_sanitization FAILED: {e}")

            try:
                self.kyc_test.test_special_characters_preservation()
                print("✅ test_special_characters_preservation PASSED")
            except Exception as e:
                print(f"❌ test_special_characters_preservation FAILED: {e}")

        def test_all_batch_integration(self):
            """Run all batch integration tests."""
            try:
                self.batch_test.test_batch_processing_mock()
                print("✅ test_batch_processing_mock PASSED")
            except Exception as e:
                print(f"❌ test_batch_processing_mock FAILED: {e}")

        def test_all_edge_cases(self):
            """Run all edge case tests."""
            try:
                self.edge_test.test_empty_input()
                print("✅ test_empty_input PASSED")
            except Exception as e:
                print(f"❌ test_empty_input FAILED: {e}")

            try:
                self.edge_test.test_unicode_handling()
                print("✅ test_unicode_handling PASSED")
            except Exception as e:
                print(f"❌ test_unicode_handling FAILED: {e}")

    # Run the tests
    print("Running comprehensive test suite...")
    print("=" * 50)

    runner = TestRunner()
    runner.setUp()

    print("\nKYC Sanitization Tests:")
    runner.test_all_kyc_sanitization()

    print("\nBatch Integration Tests:")
    runner.test_all_batch_integration()

    print("\nEdge Case Tests:")
    runner.test_all_edge_cases()

    print("\n" + "=" * 50)
    print("Test execution completed!")
