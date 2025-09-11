import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from ray.data import Dataset
import os
import json


class BatchLLMProcessor:
    def __init__(
        self,
        model_id: str,
        engine_kwargs: dict = None,
        concurrency: int = 1,
        batch_size: int = 1,
        work_dir: str = "/tmp",
    ):
        """
        Initialize the batch processor.

        Args:
            model_id: The model identifier, e.g. "meta-llama/Llama-3.1-8B-Instruct".
            engine_kwargs: Additional keyword arguments to the vLLM engine.
            concurrency: Number of concurrent batches / workers.
            batch_size: Number of items per batch.
            work_dir: Directory where outputs will be written.
        """
        self.model_id = model_id
        self.engine_kwargs = engine_kwargs or {}
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.work_dir = work_dir

        # Ensure work_dir exists
        os.makedirs(self.work_dir, exist_ok=True)

        # Ray initialization happens here
        ray.init()  # assume it finds the cluster or is local

        # Create the vLLM engine config
        self.cfg = vLLMEngineProcessorConfig(
            model=self.model_id,
            engine_kwargs=self.engine_kwargs,
            concurrency=self.concurrency,
            batch_size=self.batch_size,
        )

        self.processor = build_llm_processor(
            self.cfg,
            preprocess=self._default_preprocess,
            postprocess=self._default_postprocess,
        )

    def _default_preprocess(self, row: dict) -> dict:
        return dict(
            messages=[
                {"role": "system", "content": "You write short answers."},
                {"role": "user", "content": row["prompt"]},
            ],
            sampling_params=dict(temperature=0.0, max_tokens=128),
        )

    def _default_postprocess(self, row: dict) -> dict:
        # `row` includes generated_text and original columns
        return {"out": row["generated_text"], **row}

    def write(self, inputs: list, output_name: str = "output.json"):
        """
        Run inference on a batch of inputs and write the result.

        Args:
            inputs: a list of dicts, where each dict has at least a "prompt" key.
            output_name: name of the output file inside work_dir.
        """
        ds: Dataset = ray.data.from_items(inputs)
        # Process and write to a JSON file
        output_path = os.path.join(self.work_dir, output_name)
        self.processor(ds).write_json(output_path)
        return output_path

    def close(self):
        """Shutdown or cleanup Ray resources if needed."""
        ray.shutdown()

    def __enter__(self):
        # No extra work needed beyond __init__
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        self.close()
        # Don't suppress exceptions
        return False


# Example usage:

if __name__ == "__main__":
    inputs = [{"prompt": "Summarise AML PEP-screening risks."}]

    with BatchLLMProcessor(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_kwargs={"max_model_len": 16384},
        concurrency=1,
        batch_size=64,
        work_dir="/tmp/my_batch_results",
    ) as processor:

        out_path = processor.write(inputs, output_name="aml_summary.json")
        print(f"Results written to {out_path}")
(venv1) root@ubuntu:~/project# 
