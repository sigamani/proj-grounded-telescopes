import argparse
import json
import os
import sys
import time
import uuid
from typing import Dict, Any, List, Optional

import ray
from flask import Flask, request, jsonify
from langfuse import Langfuse

from app.kyc import KYCOrchestrator
from data.kyc_data import KYC_TEST_CASES

app = Flask(__name__)
orchestrator = None
batch_jobs: Dict[str, Dict[str, Any]] = {}  # Store batch job status and results
langfuse: Optional[Langfuse] = None

# Global session and user IDs for tracing
SESSION_ID = str(uuid.uuid4())
USER_ID = os.getenv("USER_ID", "kyc-system-user")


def init_langfuse() -> Langfuse:
    """Initialize Langfuse for tracing."""
    global langfuse

    if langfuse is None:
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )

    return langfuse


def trace_operation(name: str, input_data: Any = None, metadata: Optional[Dict[str, Any]] = None):
    """Decorator/context manager for Langfuse tracing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            lf = init_langfuse()

            trace_metadata = {
                "session_id": SESSION_ID,
                "user_id": USER_ID,
                "timestamp": time.time(),
                "operation": name,
                **(metadata or {})
            }

            try:
                # Start trace
                trace = lf.trace(
                    name=name,
                    input=input_data or {"args": args, "kwargs": kwargs},
                    metadata=trace_metadata
                )

                # Execute function
                result = func(*args, **kwargs)

                # End trace with success
                if hasattr(trace, 'end'):
                    trace.end(output=result)

                return result

            except Exception as e:
                # End trace with error
                if 'trace' in locals() and hasattr(trace, 'end'):
                    trace.end(error=str(e))
                raise

        return wrapper
    return decorator


def get_ray_config() -> Dict[str, Any]:
    """Get Ray configuration from environment variables."""
    config = {
        "dashboard_host": "0.0.0.0",
        "dashboard_port": 8265,
        "include_dashboard": True,
        "ignore_reinit_error": True,
        "log_to_driver": True
    }

    # Dynamic CPU/GPU from environment (set in docker compose)
    if num_cpus := os.getenv("RAY_NUM_CPUS"):
        config["num_cpus"] = int(num_cpus)
    if num_gpus := os.getenv("RAY_NUM_GPUS"):
        config["num_gpus"] = int(num_gpus)

    # Object store memory from environment
    if obj_mem := os.getenv("RAY_OBJECT_STORE_MEMORY"):
        config["object_store_memory"] = int(obj_mem) * 1024 * 1024 * 1024  # Convert GB to bytes
    else:
        config["object_store_memory"] = 2 * 1024 * 1024 * 1024  # Default 2GB

    return config


def init_kyc_orchestrator(ray_address: str = "auto") -> Any:
    """Initialize Ray and KYC orchestrator."""
    global orchestrator

    if orchestrator is None:
        if not ray.is_initialized():
            if ray_address == "auto":
                ray.init(**get_ray_config())
            else:
                ray.init(address=ray_address)

        orchestrator = KYCOrchestrator.remote()  # type: ignore

    return orchestrator


def validate_kyc_input(case_data: Dict[str, Any]) -> None:
    """Validate KYC input according to specification."""
    required_fields = ["business_name", "address", "country_code"]
    for field in required_fields:
        if field not in case_data:
            raise ValueError(f"Missing required field: {field}")

    # Validate country_code is ISO2 format (2 characters)
    if len(case_data["country_code"]) != 2:
        raise ValueError("country_code must be ISO2 format (2 characters)")


def process_kyc_case(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single KYC case and return structured result."""
    validate_kyc_input(case_data)

    orch = init_kyc_orchestrator()
    future = orch.process_kyc_case.remote(case_data)  # type: ignore
    result: Dict[str, Any] = ray.get(future)  # type: ignore

    return {
        "verdict": result.get("verdict", "REVIEW"),
        "structured_data": {
            "company": {
                "name": case_data["business_name"],
                "address": case_data["address"],
                "registration_id": case_data.get("registration_id"),
                "country_code": case_data["country_code"],
                "website_url": case_data.get("website_url")
            },
            "people": [],
            "industry": {},
            "online_presence": case_data.get("website_url", "")
        },
        "customer_due_diligence_report": {
            "risk_score": result.get("risk_score", 5),
            "confidence": result.get("confidence", 50),
            "reasoning": result.get("reasoning", ""),
            "audit_trail": result.get("audit_trail", []),
            "processing_time": result.get("processing_time", "")
        }
    }


@ray.remote
def process_batch_job(job_id: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ray remote function to process a batch of KYC cases."""
    results = []
    start_time = time.time()

    for i, case_data in enumerate(cases):
        try:
            result = process_kyc_case(case_data)
            results.append({
                "case_index": i,
                "case_data": case_data,
                "result": result,
                "status": "completed"
            })
        except Exception as e:
            results.append({
                "case_index": i,
                "case_data": case_data,
                "error": str(e),
                "status": "failed"
            })

    processing_time = time.time() - start_time

    return {
        "job_id": job_id,
        "batch_size": len(cases),
        "processed_cases": len(results),
        "successful_cases": len([r for r in results if r["status"] == "completed"]),
        "failed_cases": len([r for r in results if r["status"] == "failed"]),
        "results": results,
        "processing_time": f"{processing_time:.2f}s",
        "status": "completed",
        "completed_at": time.time()
    }


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint for KYC processing."""
    try:
        data = request.get_json()

        if not data or "messages" not in data:
            return jsonify({"error": "Invalid request format"}), 400

        # Extract KYC data from the last user message
        user_message = data["messages"][-1]["content"]

        try:
            # Try to parse as JSON first
            input_data = json.loads(user_message)
        except json.JSONDecodeError:
            return jsonify({"error": "Input must be valid JSON"}), 400

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Check if it's a single case or batch
        if isinstance(input_data, list):
            # Submit batch job to Ray for asynchronous processing
            batch_jobs[job_id] = {
                "status": "processing",
                "submitted_at": time.time(),
                "batch_size": len(input_data)
            }

            # Submit to Ray for processing
            future = process_batch_job.remote(job_id, input_data)  # type: ignore
            batch_jobs[job_id]["future"] = future

            response_content = json.dumps({
                "job_id": job_id,
                "status": "processing",
                "message": f"Batch job submitted with {len(input_data)} cases. Check status at /v1/jobs/{job_id}"
            }, indent=2)
        else:
            # Single case processing (synchronous)
            result = process_kyc_case(input_data)
            response_content = json.dumps(result, indent=2)

        # Format response in OpenAI chat completion format
        response = {
            "id": f"kyc-job-{job_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "kyc-compliance-v1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(user_message.split()) + len(response_content.split())
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Get the status and results of a batch job."""
    if job_id not in batch_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_info = batch_jobs[job_id]

    if job_info["status"] == "processing":
        # Check if the Ray job is complete
        try:
            if ray.is_initialized():
                future = job_info.get("future")
                if future and ray.get(future, timeout=0.1):  # type: ignore
                    result = ray.get(future)  # type: ignore
                    job_info.update(result)
                    job_info["status"] = "completed"
        except ray.exceptions.GetTimeoutError:  # type: ignore
            pass  # Still processing
        except Exception as e:
            job_info["status"] = "failed"
            job_info["error"] = str(e)

    return jsonify(job_info)


@app.route("/v1/jobs", methods=["GET"])
def list_jobs():
    """List all batch jobs."""
    jobs_summary = []
    for job_id, job_info in batch_jobs.items():
        jobs_summary.append({
            "job_id": job_id,
            "status": job_info["status"],
            "submitted_at": job_info["submitted_at"],
            "batch_size": job_info.get("batch_size", 0)
        })

    return jsonify({"jobs": jobs_summary})


@app.route("/v1/kyc/check", methods=["POST"])
def kyc_check():
    """Direct KYC check endpoint."""
    try:
        case_data = request.get_json()

        if not case_data:
            return jsonify({"error": "No JSON data provided"}), 400

        result = process_kyc_case(case_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/kyc/test-cases", methods=["GET"])
def get_test_cases():
    """Get all 10 test cases."""
    return jsonify({"test_cases": KYC_TEST_CASES})


@app.route("/v1/kyc/test/<test_id>", methods=["GET"])
def run_specific_test(test_id: str):
    """Run a specific test case by ID."""
    test_case = next((tc for tc in KYC_TEST_CASES if tc["id"] == test_id), None)

    if not test_case:
        return jsonify({"error": f"Test case {test_id} not found"}), 404

    # Remove metadata fields for processing
    case_data = {k: v for k, v in test_case.items() if k not in ["id", "description", "expected_verdict", "expected_risk"]}

    result = process_kyc_case(case_data)

    return jsonify({
        "test_case": test_case,
        "result": result,
        "expected": {
            "verdict": test_case["expected_verdict"],
            "risk": test_case["expected_risk"]
        }
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "kyc-compliance"})


def run_tests() -> None:
    """Run all 10 test cases."""
    print("Running 10 KYC test cases...")

    for test_case in KYC_TEST_CASES:
        print(f"\n=== Test Case {test_case['id']}: {test_case['description']} ===")

        # Remove metadata for processing
        case_data = {k: v for k, v in test_case.items() if k not in ["id", "description", "expected_verdict", "expected_risk"]}

        try:
            result = process_kyc_case(case_data)

            print(f"Input: {json.dumps(case_data, indent=2)}")
            print(f"Verdict: {result['verdict']}")
            print(f"Risk Score: {result['customer_due_diligence_report']['risk_score']}")
            print(f"Expected: {test_case['expected_verdict']} (risk: {test_case['expected_risk']})")

        except Exception as e:
            print(f"Error processing test case: {str(e)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KYC Compliance API Server")
    parser.add_argument("--mode", choices=["server", "test"], default="server")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.ray_address != "auto":
        os.environ["RAY_ADDRESS"] = args.ray_address

    try:
        if args.mode == "server":
            init_kyc_orchestrator(args.ray_address)
            print(f"Starting KYC API server on port {args.port}")
            app.run(host="0.0.0.0", port=args.port, debug=False)
        elif args.mode == "test":
            run_tests()

    except KeyboardInterrupt:
        if ray.is_initialized():
            ray.shutdown()
        print("\nShutdown complete")


if __name__ == "__main__":
    main()
