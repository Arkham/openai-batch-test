#!/usr/bin/env python3
"""
Helper script to create custom batch request files for OpenAI Batch API.
"""

import json
from typing import Any, Dict, List

import click
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

console = Console()


def create_chat_completion_request(
    custom_id: str,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    max_tokens: int = 150,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Create a chat completion request for batch processing.

    Args:
        custom_id: Unique identifier for the request
        messages: List of message dictionaries
        model: Model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        Formatted batch request dictionary
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    }


def create_embedding_request(
    custom_id: str, input_text: str, model: str = "text-embedding-3-small"
) -> Dict[str, Any]:
    """
    Create an embedding request for batch processing.

    Args:
        custom_id: Unique identifier for the request
        input_text: Text to embed
        model: Embedding model to use

    Returns:
        Formatted batch request dictionary
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"model": model, "input": input_text},
    }


@click.command()
@click.option("--output", "-o", default="batch_requests.jsonl", help="Output file name")
@click.option(
    "--type",
    "-t",
    type=click.Choice(["chat", "embedding", "mixed"]),
    default="chat",
    help="Type of requests to create",
)
@click.option(
    "--interactive/--no-interactive",
    "-i",
    default=False,
    help="Interactive mode to input custom prompts",
)
def main(output, type, interactive):
    """Create batch request files for OpenAI Batch API."""

    requests = []

    if interactive:
        console.print("[bold cyan]Interactive Batch Request Creator[/bold cyan]\n")

        num_requests = IntPrompt.ask(
            "How many requests do you want to create?", default=3
        )

        for i in range(num_requests):
            console.print(f"\n[yellow]Request {i + 1}:[/yellow]")

            if type == "chat" or (type == "mixed" and i % 2 == 0):
                system_msg = Prompt.ask(
                    "System message", default="You are a helpful assistant."
                )
                user_msg = Prompt.ask("User message")
                model = Prompt.ask("Model", default="gpt-4o-mini")
                max_tokens = IntPrompt.ask("Max tokens", default=150)

                request = create_chat_completion_request(
                    custom_id=f"request-{i + 1}",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    model=model,
                    max_tokens=max_tokens,
                )
            else:
                text = Prompt.ask("Text to embed")
                model = Prompt.ask("Embedding model", default="text-embedding-3-small")

                request = create_embedding_request(
                    custom_id=f"request-{i + 1}", input_text=text, model=model
                )

            requests.append(request)

    else:
        # Create sample requests based on type
        if type == "chat":
            sample_questions = [
                "What is machine learning?",
                "Explain the water cycle.",
                "How does a computer work?",
                "What is climate change?",
                "Describe the solar system.",
            ]

            for i, question in enumerate(sample_questions, 1):
                request = create_chat_completion_request(
                    custom_id=f"request-{i}",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that provides clear and concise explanations.",
                        },
                        {"role": "user", "content": question},
                    ],
                )
                requests.append(request)

        elif type == "embedding":
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Python is a popular programming language.",
                "The Earth orbits around the Sun.",
                "Water freezes at 0 degrees Celsius.",
            ]

            for i, text in enumerate(sample_texts, 1):
                request = create_embedding_request(
                    custom_id=f"embedding-{i}", input_text=text
                )
                requests.append(request)

        else:  # mixed
            # Create some chat completions
            chat_questions = [
                "What is artificial intelligence?",
                "How do neural networks work?",
                "Explain quantum computing.",
            ]

            for i, question in enumerate(chat_questions, 1):
                request = create_chat_completion_request(
                    custom_id=f"chat-{i}",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ],
                )
                requests.append(request)

            # Create some embeddings
            embedding_texts = [
                "Deep learning revolutionizes AI.",
                "Quantum computers use qubits.",
            ]

            for i, text in enumerate(embedding_texts, 1):
                request = create_embedding_request(
                    custom_id=f"embedding-{i}", input_text=text
                )
                requests.append(request)

    # Write requests to file
    with open(output, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")

    console.print(
        f"\n[green]✓[/green] Created {len(requests)} batch requests in '{output}'"
    )

    # Display summary
    console.print("\n[bold]Request Summary:[/bold]")
    for i, request in enumerate(requests[:3], 1):
        console.print(f"  {i}. {request['custom_id']} - {request['url']}")

    if len(requests) > 3:
        console.print(f"  ... and {len(requests) - 3} more")


if __name__ == "__main__":
    main()
