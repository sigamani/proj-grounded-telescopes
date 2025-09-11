"""
Ray Jobs API client for submitting batch inference jobs.
The role of this script is to submits jobs to the Ray 
cluster via the Jobs REST API. Used by the jobs-runner 
service in production deployment.
"""
import requests
import json
import time
import sys
import os
from pathlib import Path


class RayJobsClient:
    def __init__(self, ray_head_url="http://ray-head:8265"):
        self.ray_head_url = ray_head_url
        self.jobs_api_url = f"{ray_head_url}/api/jobs/"
    
    def submit_job(self, entrypoint, runtime_env=None, metadata=None):
        """Submit a job to Ray cluster."""
        job_spec = {
            "entrypoint": entrypoint,
            "job_id": None,  
            "runtime_env": runtime_env or {},
            "metadata": metadata or {}
        }
        
        print(f"Submitting job to {self.jobs_api_url}")
        print(f"Job spec: {json.dumps(job_spec, indent=2)}")
        
        try:
            response = requests.post(
                self.jobs_api_url,
                json=job_spec,
                timeout=30
            )
            response.raise_for_status()
            
            job_info = response.json()
            job_id = job_info["job_id"]
            
            print(f"Job submitted successfully!")
            print(f"Job ID: {job_id}")
            print(f"Status: {job_info.get('status', 'UNKNOWN')}")
            
            return job_id
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to submit job: {e}")
            return None
    
    def get_job_status(self, job_id):
        """Get status of a specific job."""
        try:
            response = requests.get(f"{self.jobs_api_url}{job_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get job status: {e}")
            return None
    
    def wait_for_job_completion(self, job_id, timeout=300):
        """Wait for job to complete."""
        print(f"Waiting for job {job_id} to complete...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            job_status = self.get_job_status(job_id)
            
            if not job_status:
                time.sleep(5)
                continue
            
            status = job_status.get("status", "UNKNOWN")
            print(f"Job status: {status}")
            
            if status == "SUCCEEDED":
                print("Job completed successfully!")
                return True
            elif status == "FAILED":
                print("Job failed!")
                print(f"Error details: {job_status.get('message', 'No details available')}")
                return False
            elif status in ["STOPPED", "CANCELLED"]:
                print(f"Job {status.lower()}")
                return False
            
            time.sleep(10)  # Check every 10 seconds
        
        print("Timeout waiting for job completion")
        return False
    
    def list_jobs(self):
        """List all jobs."""
        try:
            response = requests.get(self.jobs_api_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to list jobs: {e}")
            return None


def run_batch_inference_job():
    """Submit the batch inference job."""
    client = RayJobsClient()
    
    # Job configuration
    entrypoint = "python src/batch_infer.py"
    runtime_env = {
        "working_dir": ".",
        "pip": [
            "ray[data]==2.49.1", 
            "vllm==0.10.0",
            "torch"
        ]
    }
    metadata = {
        "job_type": "batch_inference",
        "version": "0.1.1"
    }
    
    # Submit job
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        metadata=metadata
    )
    
    if not job_id:
        sys.exit(1)
    
    # Wait for completion
    success = client.wait_for_job_completion(job_id, timeout=600)  # 10 minutes
    
    if success:
        print("🎉 Batch inference completed successfully!")
        sys.exit(0)
    else:
        print("💥 Batch inference failed!")
        sys.exit(1)


def run_custom_job(script_path):
    """Run a custom Python script as a Ray job."""
    if not Path(script_path).exists():
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)
    
    client = RayJobsClient()
    
    entrypoint = f"python {script_path}"
    runtime_env = {"working_dir": "."}
    metadata = {"job_type": "custom", "script": script_path}
    
    job_id = client.submit_job(entrypoint, runtime_env, metadata)
    
    if job_id:
        client.wait_for_job_completion(job_id)
    else:
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Custom script provided
        script_path = sys.argv[1]
        run_custom_job(script_path)
    else:
        # Default: run batch inference
        run_batch_inference_job()


if __name__ == "__main__":
    main()
