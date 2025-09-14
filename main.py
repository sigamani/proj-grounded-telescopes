import argparse
import os
import sys

import ray

from data.kyc_data import FALSE_POSITIVE_CASES, TRUE_POSITIVE_CASES
from app.kyc import KYCOrchestrator, create_kyc_batch_processor
from tests.kyc_tests import (
    test_batch_processing,
    test_enhanced_kyc_pipeline,
    test_kyc_pipeline,
    test_uk_kyc_dataset,
)


def run_kyc_server(ray_address="auto"):
    if not ray.is_initialized():
        if ray_address == "auto":
            # Local development mode
            ray.init(
                num_cpus=4,
                num_gpus=0,
                object_store_memory=2*1024*1024*1024,  # 2GB object store
                dashboard_host="0.0.0.0",
                dashboard_port=8265,
                include_dashboard=True,
                ignore_reinit_error=True,
                log_to_driver=True
            )
        else:
            ray.init(address=ray_address)
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


def setup_graphrag():
    from app.preprocessing import main as setup
    return setup()


def main():
    parser = argparse.ArgumentParser(description="KYC GenAI-Native Compliance System")
    parser.add_argument(
        "--mode",
        choices=["server", "test", "batch", "setup"],
        default="server",
        help="Run mode: server, test, batch inference, or setup GraphRAG",
    )
    parser.add_argument("--ray-address", default="auto", help="Ray cluster address")

    args = parser.parse_args()

    if args.ray_address != "auto":
        os.environ["RAY_ADDRESS"] = args.ray_address

    try:
        if args.mode == "server":
            run_kyc_server(args.ray_address)
        elif args.mode == "test":
            run_tests()
        elif args.mode == "batch":
            run_batch_inference()
        elif args.mode == "setup":
            setup_graphrag()

    except KeyboardInterrupt:
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
