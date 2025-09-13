#!/usr/bin/env python3
"""
Simple test script to validate vLLM batch inference through HTTP API.
"""
import json
import requests
import sys

def test_vllm_batch_inference():
    """Test the vLLM server with multiple requests."""

    # Test data - batch of prompts
    test_prompts = [
        "Summarise AML PEP-screening risks.",
        "What are the main compliance challenges?",
        "Explain the key features of KYC processes."
    ]

    api_url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    print("Testing vLLM batch inference...")
    print(f"API URL: {api_url}")
    print(f"Number of prompts: {len(test_prompts)}")
    print("-" * 50)

    results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nProcessing prompt {i}/{len(test_prompts)}: {prompt[:50]}...")

        payload = {
            "model": "qwen-1.5b",
            "messages": [
                {"role": "system", "content": "You write short answers."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 128,
            "temperature": 0.2
        }

        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                results.append({
                    "prompt": prompt,
                    "response": answer,
                    "status": "success"
                })
                print(f"✓ Success: {answer[:100]}...")

            else:
                error_msg = f"API Error: {response.status_code}"
                results.append({
                    "prompt": prompt,
                    "response": error_msg,
                    "status": "error"
                })
                print(f"✗ Error: {error_msg}")

        except Exception as e:
            error_msg = f"Request failed: {e}"
            results.append({
                "prompt": prompt,
                "response": error_msg,
                "status": "error"
            })
            print(f"✗ Exception: {error_msg}")

    print("\n" + "="*60)
    print("BATCH INFERENCE TEST RESULTS")
    print("="*60)

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count

    print(f"Total requests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Success rate: {success_count/len(results)*100:.1f}%")

    if success_count > 0:
        print("\nSample responses:")
        for i, result in enumerate(results[:3], 1):
            if result["status"] == "success":
                print(f"\n{i}. Prompt: {result['prompt']}")
                print(f"   Response: {result['response'][:200]}...")

    # Save results to file
    output_file = "/tmp/batch_inference_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return success_count == len(results)

if __name__ == "__main__":
    success = test_vllm_batch_inference()

    if success:
        print("\n🎉 All batch inference tests passed!")
        print("✅ vLLM server is working correctly with Qwen/Qwen2.5-1.5B-Instruct model")
        sys.exit(0)
    else:
        print("\n❌ Some batch inference tests failed!")
        sys.exit(1)