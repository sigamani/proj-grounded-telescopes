import asyncio
import ray
from typing import List, Dict, Any

from .models import KYCInput, KYCResult
from .agents import IdentityAgent, ScreeningAgent

@ray.remote
class KYCOrchestrator:

    def __init__(self):
        self.identity_agent = IdentityAgent("identity-verification")
        self.screening_agent = ScreeningAgent("sanctions-screening")

    async def process_kyc_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        kyc_input = KYCInput(**case_data)
        context = {}
        audit_trail = []

        identity_result = await self.identity_agent.process(kyc_input, context)
        context.update(identity_result)
        audit_trail.append(f"Identity: {identity_result['reasoning']}")

        screening_result = await self.screening_agent.process(kyc_input, context)
        audit_trail.append(f"Screening: {screening_result['reasoning']}")

        risk_score = min(10, 3 + len(screening_result.get("sanctions_matches", [])) * 2)
        verdict = "REJECT" if risk_score >= 8 else "REVIEW" if risk_score >= 5 else "ACCEPT"

        return {
            "verdict": verdict,
            "risk_score": risk_score,
            "confidence": 0.8,
            "reasoning": f"Risk {risk_score}/10 based on {len(screening_result.get('sanctions_matches', []))} sanctions matches",
            "findings": {
                "identity": identity_result,
                "screening": screening_result
            },
            "audit_trail": audit_trail
        }


def create_batch_processor():
    """Create a batch processor for LLM inference using Ray Data."""

    if not ray.is_initialized():
        ray.init(address="ray://localhost:10001")

    def process_batch_function(batch):
        """Function to process batch via Ray Data."""
        processor = VLLMAPIProcessor.remote()
        future = processor.process_batch.remote(batch)
        return ray.get(future)

    def ray_data_processor(ds):
        """Process Ray dataset using map_batches."""
        preprocessed_ds = ds  # data sanitisation
        result_ds = preprocessed_ds.map_batches(process_batch_function, batch_size=2)
        return result_ds  # agent triage

    return ray_data_processor


def run_batch_inference():
    """Run batch inference if called directly."""
    processor = create_batch_processor()
    if processor is None:
        print("Processor not available - skipping inference")
        return

    ds = ray.data.from_items([{"prompt": "Summarise AML PEP-screening risks."}])
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = f"{temp_dir}/out"
        processor(ds).write_json(output_path)


if __name__ == "__main__":
    run_batch_inference()
