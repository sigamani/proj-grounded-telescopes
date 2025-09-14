#!/usr/bin/env python3
import argparse
import os
import sys

import ray

from data.kyc_data import FALSE_POSITIVE_CASES, TRUE_POSITIVE_CASES
from src.kyc import KYCOrchestrator, create_kyc_batch_processor
from tests.kyc_tests import (
    test_batch_processing,
    test_enhanced_kyc_pipeline,
    test_kyc_pipeline,
    test_uk_kyc_dataset,
)


def run_kyc_server():
    if not ray.is_initialized():
        ray.init()
    orchestrator = KYCOrchestrator.remote()
    return orchestrator


def run_tests():
    test_kyc_pipeline()
    test_enhanced_kyc_pipeline()
    test_batch_processing()
    test_uk_kyc_dataset()


def run_batch_inference():
    create_kyc_batch_processor()
    orchestrator = run_kyc_server()
    test_cases = TRUE_POSITIVE_CASES + FALSE_POSITIVE_CASES
    futures = [orchestrator.process_kyc_case.remote(case) for case in test_cases]
    return ray.get(futures)


def main():
    parser = argparse.ArgumentParser(description="KYC GenAI-Native Compliance System")
    parser.add_argument(
        "--mode",
        choices=["server", "test", "batch"],
        default="server",
        help="Run mode: server, test, or batch inference",
    )
    parser.add_argument("--ray-address", default="auto", help="Ray cluster address")

    args = parser.parse_args()

    if args.ray_address != "auto":
        os.environ["RAY_ADDRESS"] = args.ray_address

    try:
        if args.mode == "server":
            run_kyc_server()
        elif args.mode == "test":
            run_tests()
        elif args.mode == "batch":
            run_batch_inference()

    except KeyboardInterrupt:
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
