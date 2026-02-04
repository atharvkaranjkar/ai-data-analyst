import boto3
import json
import os


def ask_aws_llm(
    prompt: str,
    model_id: str = "amazon.nova-lite-v1:0",
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> dict:

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

    body = {
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ],
            }
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.9,
            "topK": 20,
        },
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )

    model_response = json.loads(response["body"].read())
    content_blocks = model_response.get("output", {}).get("message", {}).get("content", [])
    response_text = next(
        (block.get("text") for block in content_blocks if isinstance(block, dict) and "text" in block),
        None,
    )
    
    usage = model_response.get("usage", {})
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    
    return {
        "text": response_text if response_text else "",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
