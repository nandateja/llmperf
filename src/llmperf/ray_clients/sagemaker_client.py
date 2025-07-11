import io
import json
import os
import time
from typing import Any, Dict

import boto3
import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class SageMakerClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def __init__(self):
        # Sagemaker doesn't return the number of tokens that are generated so we approximate it by
        # using the llama tokenizer.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            raise ValueError("AWS_ACCESS_KEY_ID must be set.")
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            raise ValueError("AWS_SECRET_ACCESS_KEY must be set.")
        if not os.environ.get("AWS_REGION_NAME"):
            raise ValueError("AWS_REGION_NAME must be set.")

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        model = request_config.model
        sm_runtime = boto3.client(
            "sagemaker-runtime", region_name=os.environ.get("AWS_REGION_NAME")
        )

        sampling_params = request_config.sampling_params

        if "max_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = sampling_params["max_tokens"]
            del sampling_params["max_tokens"]

        message = {
            "inputs": prompt,
            "parameters": {
                **request_config.sampling_params,
            },
            "stream": True,
        }

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            response = sm_runtime.invoke_endpoint_with_response_stream(
                EndpointName=model,
                ContentType="application/json",
                Body=json.dumps(message),
                CustomAttributes="accept_eula=true",
            )
            
            event_stream = response["Body"]

            generated_text = ""
            start_json = b'{'
            tokens_received = 0

            for line, line_ttft, _ in LineIterator(event_stream):
                if line != b'' and start_json in line:
                    json_start = line.find(start_json)
                    json_data = line[json_start:].decode('utf-8')
                    data = json.loads(json_data)
                    if 'token' in data and 'text' in data['token']:
                        token_text = data['token']['text']
                        generated_text += token_text
                        tokens_received += 1

                        if tokens_received == 1:
                            # First token - use LineIterator's TTFT
                            ttft = line_ttft - start_time
                            time_to_next_token.append(ttft)
                        else:
                            # Subsequent tokens - calculate inter-token latency
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
            total_request_time = time.monotonic() - start_time

            if generated_text:
                # Use the tokenizer to get accurate token count
                tokens_received = len(self.tokenizer.encode(generated_text))
                output_throughput = tokens_received / total_request_time if total_request_time > 0 else 0
            else:
                print("No generated text found in stream")

        except Exception as e:
            error_msg = str(e)
            error_response_code = 500
            print(f"Warning Or Error: {e}")
            print(error_response_code)
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config


class LineIterator:
    """
    A helper class for parsing the byte stream input.
    Reference: https://aws.amazon.com/blogs/machine-learning/elevating-the-generative-ai-experience-introducing-streaming-support-in-amazon-sagemaker-hosting/
    """

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0
        self.ttft = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                if self.ttft == 0:
                    self.ttft = time.monotonic()
                self.read_pos += len(line)
                return line[:-1], self.ttft, time.monotonic()
            # kyle: dealing with last ']' for chat output
            if line and self.read_pos == self.buffer.getbuffer().nbytes - 1:
                self.read_pos += 1
                return line, self.ttft, time.monotonic()
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])
