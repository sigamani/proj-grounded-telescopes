import ray
from src.kyc_utils import sanitize_kyc_input


def create_batch_processor():
    """Create a batch processor for LLM inference."""
    try:
        from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    except ImportError:
        print("vLLM not available - using mock processor for testing")
        return None

    if not ray.is_initialized():
        ray.init()

    cfg = vLLMEngineProcessorConfig(
        model_source="microsoft/DialoGPT-medium",
        engine_kwargs={"max_model_len": 1024},
        concurrency=1,
        batch_size=32,
    )

    processor = build_llm_processor(
        cfg,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You write short answers."},
                {"role": "user", "content": row["prompt"]},
            ],
            sampling_params=dict(temperature=0.2, max_tokens=128),
        ),
        postprocess=lambda row: {"out": row["generated_text"], **row},
    )
    return processor


def run_batch_inference():
    """Run batch inference with KYC sanitization for one input job."""
    processor = create_batch_processor()
    if processor is None:
        print("Processor not available - skipping inference")
        return

    # Sample KYC data for sanitization
    sample_kyc_data = {
        "business_name": "   Acme Financial Services Ltd   ",
        "address": "  123 Main Street, London, UK  ",
        "country_code": "gb",
        "registration_id": "  12345678  ",
        "website_url": "acmefinance.com",
        "prompt": "Summarise AML PEP-screening risks.",
    }

    # Apply KYC sanitization to the input
    print("Original data:", sample_kyc_data)
    sanitized_data = sanitize_kyc_input(sample_kyc_data)
    print("Sanitized data:", sanitized_data)

    # Create dataset with sanitized data
    ds = ray.data.from_items([sanitized_data])
    processor(ds).write_json("/tmp/out")

    print("Batch inference completed with KYC sanitization applied")


if __name__ == "__main__":
    run_batch_inference()
