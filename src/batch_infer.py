import ray
import requests


@ray.remote
class VLLMAPIProcessor:
    """Ray actor for processing requests via VLLM HTTP API."""
    
    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions"):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}
    
    def process_batch(self, batch: list) -> list:
        """Process a batch of prompts via HTTP API."""
        results = []
        
        for item in batch:
            payload = {
                "model": "microsoft/DialoGPT-medium",  # Match the model in compose.dev.yaml
                "messages": [
                    {"role": "system", "content": "You write short answers."},
                    {"role": "user", "content": item["prompt"]}
                ],
                "max_tokens": 128,
                "temperature": 0.2
            }
            
            try:
                response = requests.post(
                    self.api_url, 
                    json=payload, 
                    headers=self.headers, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "prompt": item["prompt"],
                        "out": result['choices'][0]['message']['content']
                    })
                else:
                    results.append({
                        "prompt": item["prompt"],
                        "out": f"API Error: {response.status_code}"
                    })
                    
            except Exception as e:
                results.append({
                    "prompt": item["prompt"],
                    "out": f"Request failed: {e}"
                })
        
        return results


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
        result_ds = ds.map_batches(process_batch_function, batch_size=2)
        return result_ds
    
    return ray_data_processor


def run_batch_inference():
    """Run batch inference if called directly."""
    processor = create_batch_processor()
    if processor is None:
        print("Processor not available - skipping inference")
        return

    ds = ray.data.from_items([{"prompt": "Summarise AML PEP-screening risks."}])
    processor(ds).write_json("/tmp/out")


if __name__ == "__main__":
    run_batch_inference()
