#!/usr/bin/env python3
"""
OpenAI Batch API Processor

This module provides functionality to process batch requests using the OpenAI Batch API.
It supports uploading batch files, creating batch jobs, monitoring progress, and retrieving results.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv
from openai import OpenAI
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()


class BatchProcessor:
    """Handles OpenAI Batch API operations."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the BatchProcessor with OpenAI credentials.

        Args:
            api_key: OpenAI API key (optional, defaults to 'dummy' for proxy setups)
            base_url: OpenAI API base URL (defaults to OPENAI_BASE_URL or OPENAI_API_BASE env var)
        """
        # For proxy setups, API key might not be needed
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def create_batch_input(
        self, requests: List[Dict[str, Any]], output_file: str
    ) -> str:
        """
        Create a JSONL batch input file from a list of requests.

        Args:
            requests: List of request dictionaries
            output_file: Path to save the JSONL file

        Returns:
            Path to the created file
        """
        with open(output_file, "w") as f:
            for i, request in enumerate(requests):
                # Ensure each request has required fields
                batch_request = {
                    "custom_id": request.get("custom_id", f"request-{i + 1}"),
                    "method": request.get("method", "POST"),
                    "url": request.get("url", "/v1/chat/completions"),
                    "body": request.get("body", {}),
                }
                f.write(json.dumps(batch_request) + "\n")

        console.print(f"[green]✓[/green] Created batch input file: {output_file}")
        return output_file

    def upload_file(self, file_path: str) -> str:
        """
        Upload a file to OpenAI for batch processing.

        Args:
            file_path: Path to the file to upload

        Returns:
            File ID of the uploaded file
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Uploading file...", total=None)

            with open(file_path, "rb") as f:
                file_obj = self.client.files.create(file=f, purpose="batch")

            progress.update(task, completed=True)

        console.print(
            f"[green]✓[/green] File uploaded successfully. ID: [cyan]{file_obj.id}[/cyan]"
        )
        return file_obj.id

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a batch processing job.

        Args:
            input_file_id: ID of the uploaded input file
            endpoint: API endpoint for the batch
            completion_window: Time window for batch completion
            metadata: Optional metadata for the batch

        Returns:
            Batch ID
        """
        batch_params = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }

        if metadata:
            batch_params["metadata"] = metadata

        batch = self.client.batches.create(**batch_params)

        console.print(
            f"[green]✓[/green] Batch created successfully. ID: [cyan]{batch.id}[/cyan]"
        )
        console.print(f"  Status: [yellow]{batch.status}[/yellow]")
        console.print(f"  Endpoint: {batch.endpoint}")
        console.print(f"  Completion window: {batch.completion_window}")

        return batch.id

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch job.

        Args:
            batch_id: ID of the batch job

        Returns:
            Batch status information
        """
        batch = self.client.batches.retrieve(batch_id)

        # Convert request_counts object to dict if it exists
        request_counts = None
        if batch.request_counts:
            request_counts = {
                "total": getattr(batch.request_counts, "total", 0),
                "completed": getattr(batch.request_counts, "completed", 0),
                "failed": getattr(batch.request_counts, "failed", 0),
            }

        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "request_counts": request_counts,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }

    def monitor_batch(self, batch_id: str, poll_interval: int = 10) -> Dict[str, Any]:
        """
        Monitor a batch job until completion.

        Args:
            batch_id: ID of the batch job
            poll_interval: Seconds between status checks

        Returns:
            Final batch status
        """
        console.print(f"\n[bold]Monitoring batch: {batch_id}[/bold]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing batch...", total=None)

            while True:
                status = self.get_batch_status(batch_id)

                # Update progress description
                desc = f"Status: {status['status']}"
                if status["request_counts"]:
                    counts = status["request_counts"]
                    desc += f" | Completed: {counts.get('completed', 0)}/{counts.get('total', 0)}"
                    if counts.get("failed", 0) > 0:
                        desc += f" | Failed: {counts.get('failed', 0)}"

                progress.update(task, description=desc)

                if status["status"] in ["completed", "failed", "expired", "cancelled"]:
                    progress.stop()
                    break

                time.sleep(poll_interval)

        # Display final status
        self._display_batch_status(status)

        return status

    def _display_batch_status(self, status: Dict[str, Any]):
        """Display batch status in a formatted table."""
        table = Table(
            title="Batch Status", show_header=True, header_style="bold magenta"
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("ID", status["id"])
        table.add_row("Status", status["status"])
        table.add_row("Created At", str(datetime.fromtimestamp(status["created_at"])))

        if status["completed_at"]:
            table.add_row(
                "Completed At", str(datetime.fromtimestamp(status["completed_at"]))
            )

        if status["request_counts"]:
            counts = status["request_counts"]
            table.add_row("Total Requests", str(counts.get("total", 0)))
            table.add_row("Completed", str(counts.get("completed", 0)))
            table.add_row("Failed", str(counts.get("failed", 0)))

        console.print(table)

    def download_results(
        self, batch_id: str, output_dir: str = "."
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Download batch results and error files.

        Args:
            batch_id: ID of the batch job
            output_dir: Directory to save the files

        Returns:
            Tuple of (output_file_path, error_file_path)
        """
        status = self.get_batch_status(batch_id)

        output_file_path = None
        error_file_path = None

        # Download output file
        if status["output_file_id"]:
            output_file_path = os.path.join(output_dir, f"{batch_id}_output.jsonl")
            content = self.client.files.content(status["output_file_id"])

            with open(output_file_path, "wb") as f:
                f.write(content.content)

            console.print(
                f"[green]✓[/green] Downloaded output file: {output_file_path}"
            )

        # Download error file if exists
        if status["error_file_id"]:
            error_file_path = os.path.join(output_dir, f"{batch_id}_errors.jsonl")
            content = self.client.files.content(status["error_file_id"])

            with open(error_file_path, "wb") as f:
                f.write(content.content)

            console.print(
                f"[yellow]⚠[/yellow] Downloaded error file: {error_file_path}"
            )

        return output_file_path, error_file_path

    def parse_results(self, results_file: str) -> List[Dict[str, Any]]:
        """
        Parse batch results from a JSONL file.

        Args:
            results_file: Path to the results JSONL file

        Returns:
            List of parsed results
        """
        results = []
        with open(results_file, "r") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        return results

    def list_batches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent batch jobs.

        Args:
            limit: Maximum number of batches to retrieve

        Returns:
            List of batch information
        """
        batches = self.client.batches.list(limit=limit)
        return [
            {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "endpoint": batch.endpoint,
            }
            for batch in batches.data
        ]

    def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel a batch job.

        Args:
            batch_id: ID of the batch to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            self.client.batches.cancel(batch_id)
            console.print(f"[green]✓[/green] Batch {batch_id} cancelled successfully")
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to cancel batch: {e}")
            return False


@click.group()
def cli():
    """OpenAI Batch API Processor CLI"""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--endpoint", default="/v1/chat/completions", help="API endpoint for the batch"
)
@click.option("--window", default="24h", help="Completion window (e.g., 24h)")
@click.option(
    "--monitor/--no-monitor", default=True, help="Monitor batch until completion"
)
@click.option(
    "--download/--no-download", default=True, help="Download results when completed"
)
@click.option("--output-dir", default=".", help="Directory to save results")
def process(input_file, endpoint, window, monitor, download, output_dir):
    """Process a batch input file."""
    try:
        processor = BatchProcessor()

        # Upload file
        file_id = processor.upload_file(input_file)

        # Create batch
        batch_id = processor.create_batch(file_id, endpoint, window)

        # Monitor if requested
        if monitor:
            status = processor.monitor_batch(batch_id)

            # Download results if completed and requested
            if download and status["status"] == "completed":
                output_file, error_file = processor.download_results(
                    batch_id, output_dir
                )

                if output_file:
                    # Parse and display summary
                    results = processor.parse_results(output_file)
                    console.print(f"\n[bold]Results Summary:[/bold]")
                    console.print(f"Total responses: {len(results)}")

                    # Show first few results
                    for i, result in enumerate(results[:3], 1):
                        console.print(f"\n[cyan]Response {i}:[/cyan]")
                        console.print(f"Custom ID: {result.get('custom_id')}")
                        if "response" in result and "body" in result["response"]:
                            body = result["response"]["body"]
                            if "choices" in body and body["choices"]:
                                message = body["choices"][0].get("message", {})
                                console.print(
                                    f"Content: {message.get('content', 'N/A')[:200]}..."
                                )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--limit", default=10, help="Number of batches to list")
def list(limit):
    """List recent batch jobs."""
    try:
        processor = BatchProcessor()
        batches = processor.list_batches(limit)

        if not batches:
            console.print("No batches found.")
            return

        table = Table(
            title="Recent Batches", show_header=True, header_style="bold magenta"
        )
        table.add_column("ID", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Created At", style="green")
        table.add_column("Endpoint", style="white")

        for batch in batches:
            created_at = datetime.fromtimestamp(batch["created_at"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            table.add_row(batch["id"], batch["status"], created_at, batch["endpoint"])

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("batch_id")
def status(batch_id):
    """Get status of a specific batch job."""
    try:
        processor = BatchProcessor()
        status = processor.get_batch_status(batch_id)
        processor._display_batch_status(status)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("batch_id")
@click.option("--output-dir", default=".", help="Directory to save results")
def download(batch_id, output_dir):
    """Download results for a completed batch."""
    try:
        processor = BatchProcessor()
        output_file, error_file = processor.download_results(batch_id, output_dir)

        if output_file:
            results = processor.parse_results(output_file)
            console.print(f"Downloaded {len(results)} results")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("batch_id")
def cancel(batch_id):
    """Cancel a batch job."""
    try:
        processor = BatchProcessor()
        processor.cancel_batch(batch_id)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
