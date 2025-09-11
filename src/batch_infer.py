import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

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

ds = ray.data.from_items([{"prompt": "Summarise AML PEP-screening risks."}])
processor(ds).write_json("/tmp/out")
