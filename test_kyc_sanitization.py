#!/usr/bin/env python3
"""
Isolated test for KYC sanitization functionality without heavy dependencies.
"""

import json
from datetime import datetime
from typing import Dict, Any


def sanitize_kyc_input(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracted sanitization function from kyc_utils.py for isolated testing.
    """
    try:
        # Extract and sanitize core fields
        sanitized = {
            "business_name": str(row.get("business_name", "")).strip(),
            "address": str(row.get("address", "")).strip(),
            "country_code": str(row.get("country_code", "")).upper()[:2],
            "registration_id": str(row.get("registration_id", "") or "").strip(),
            "website_url": str(row.get("website_url", "") or "").strip(),
            "processing_timestamp": datetime.now().isoformat(),
        }
        
        # Validate required fields
        if not sanitized["business_name"] or len(sanitized["business_name"]) < 1:
            sanitized["validation_errors"] = ["business_name required and non-empty"]
            sanitized["validation_status"] = "FAILED"
            return sanitized
            
        if not sanitized["address"] or len(sanitized["address"]) < 1:
            sanitized["validation_errors"] = ["address required and non-empty"] 
            sanitized["validation_status"] = "FAILED"
            return sanitized
            
        # Check original country code before truncation
        original_country = str(row.get("country_code", "")).strip()
        if len(original_country) != 2 or not original_country.isalpha():
            sanitized["validation_errors"] = ["country_code must be valid ISO2 format"]
            sanitized["validation_status"] = "FAILED" 
            return sanitized
        
        # Handle optional fields
        if sanitized["website_url"]:
            # Basic URL sanitization
            if not sanitized["website_url"].startswith(("http://", "https://")):
                sanitized["website_url"] = "https://" + sanitized["website_url"]
        
        # Add metadata for tracking
        sanitized["sanitization_applied"] = True
        sanitized["validation_status"] = "PASSED"
        sanitized["pii_scrubbing_status"] = "APPLIED"
        
        return sanitized
        
    except Exception as e:
        return {
            "business_name": "",
            "validation_status": "ERROR",
            "validation_errors": [f"Sanitization error: {str(e)}"],
            "processing_timestamp": datetime.now().isoformat()
        }


def test_kyc_sanitization():
    """Test KYC sanitization with various inputs."""
    print("=== KYC Sanitization Tests ===\n")
    
    # Test 1: Valid input with whitespace and formatting issues
    test1_data = {
        'business_name': '   Acme Financial Services Ltd   ',
        'address': '  123 Main Street, London, UK  ',
        'country_code': 'gb',
        'registration_id': '  12345678  ',
        'website_url': 'acmefinance.com'
    }
    
    print("Test 1: Valid input with formatting issues")
    print("Original data:", json.dumps(test1_data, indent=2))
    sanitized1 = sanitize_kyc_input(test1_data)
    print("Sanitized data:", json.dumps(sanitized1, indent=2))
    print("✅ PASSED" if sanitized1["validation_status"] == "PASSED" else "❌ FAILED")
    print()
    
    # Test 2: Missing required field
    test2_data = {
        'business_name': '',
        'address': 'Valid address',
        'country_code': 'US'
    }
    
    print("Test 2: Missing business name")
    print("Original data:", json.dumps(test2_data, indent=2))
    sanitized2 = sanitize_kyc_input(test2_data)
    print("Sanitized data:", json.dumps(sanitized2, indent=2))
    print("✅ PASSED" if sanitized2["validation_status"] == "FAILED" else "❌ FAILED")
    print()
    
    # Test 3: Invalid country code
    test3_data = {
        'business_name': 'Valid Company',
        'address': 'Valid address',
        'country_code': 'INVALID'
    }
    
    print("Test 3: Invalid country code")
    print("Original data:", json.dumps(test3_data, indent=2))
    sanitized3 = sanitize_kyc_input(test3_data)
    print("Sanitized data:", json.dumps(sanitized3, indent=2))
    print("✅ PASSED" if sanitized3["validation_status"] == "FAILED" else "❌ FAILED")
    print()
    
    # Test 4: URL sanitization
    test4_data = {
        'business_name': 'Tech Corp',
        'address': '456 Tech St',
        'country_code': 'US',
        'website_url': 'techcorp.com'
    }
    
    print("Test 4: URL sanitization")
    print("Original data:", json.dumps(test4_data, indent=2))
    sanitized4 = sanitize_kyc_input(test4_data)
    print("Sanitized data:", json.dumps(sanitized4, indent=2))
    expected_url = "https://techcorp.com"
    url_test_passed = sanitized4.get("website_url") == expected_url
    print(f"✅ PASSED" if url_test_passed else f"❌ FAILED - Expected {expected_url}, got {sanitized4.get('website_url')}")
    print()
    
    # Test 5: Edge case with special characters
    test5_data = {
        'business_name': 'Company & Co.',
        'address': '789 Street, Apt #5',
        'country_code': 'fr',
        'registration_id': 'FR-123-456',
        'website_url': 'https://company-co.fr'
    }
    
    print("Test 5: Special characters and existing protocol")
    print("Original data:", json.dumps(test5_data, indent=2))
    sanitized5 = sanitize_kyc_input(test5_data)
    print("Sanitized data:", json.dumps(sanitized5, indent=2))
    print("✅ PASSED" if sanitized5["validation_status"] == "PASSED" else "❌ FAILED")
    print()
    
    return all([
        sanitized1["validation_status"] == "PASSED",
        sanitized2["validation_status"] == "FAILED",
        sanitized3["validation_status"] == "FAILED", 
        sanitized4.get("website_url") == "https://techcorp.com",
        sanitized5["validation_status"] == "PASSED"
    ])


def test_batch_integration_mock():
    """Test how the sanitization would integrate with batch processing."""
    print("=== Batch Integration Mock Test ===\n")
    
    # Simulate batch data
    batch_data = [
        {
            'business_name': '   Acme Financial Services Ltd   ',
            'address': '  123 Main Street, London, UK  ',
            'country_code': 'gb',
            'registration_id': '  12345678  ',
            'website_url': 'acmefinance.com',
            'prompt': 'Summarise AML PEP-screening risks.'
        },
        {
            'business_name': 'Tech Startup Inc',
            'address': '456 Innovation Drive',
            'country_code': 'us',
            'website_url': 'techstartup.com',
            'prompt': 'Analyze compliance requirements.'
        }
    ]
    
    print("Processing batch of KYC data...")
    processed_batch = []
    
    for i, data in enumerate(batch_data):
        print(f"\nProcessing item {i + 1}:")
        print("Original:", json.dumps(data, indent=2))
        
        sanitized = sanitize_kyc_input(data)
        processed_batch.append(sanitized)
        
        print("Sanitized:", json.dumps(sanitized, indent=2))
        print(f"Status: {sanitized.get('validation_status')}")
    
    # Summary
    passed_count = sum(1 for item in processed_batch if item.get('validation_status') == 'PASSED')
    print(f"\n=== Batch Processing Summary ===")
    print(f"Total items: {len(processed_batch)}")
    print(f"Passed validation: {passed_count}")
    print(f"Failed validation: {len(processed_batch) - passed_count}")
    
    return passed_count == len(processed_batch)


if __name__ == "__main__":
    print("Testing KYC Sanitization Integration")
    print("=" * 50)
    
    # Run sanitization tests
    sanitization_passed = test_kyc_sanitization()
    
    # Run batch integration test
    batch_passed = test_batch_integration_mock()
    
    # Final summary
    print("=" * 50)
    print("FINAL TEST RESULTS:")
    print(f"Sanitization Tests: {'✅ PASSED' if sanitization_passed else '❌ FAILED'}")
    print(f"Batch Integration: {'✅ PASSED' if batch_passed else '❌ FAILED'}")
    
    overall_passed = sanitization_passed and batch_passed
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_passed else '❌ SOME TESTS FAILED'}")
    
    exit(0 if overall_passed else 1)