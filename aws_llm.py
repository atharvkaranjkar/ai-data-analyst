from unittest import result
import boto3
import json
import os


def ask_aws_llm(
    prompt: str,
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens: int = 800,
    temperature: float = 0.2
) -> str:
    """
    Send prompt to AWS Bedrock and return plain text response.
    """

    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

    body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            }


    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

