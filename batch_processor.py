#!/usr/bin/env python3
"""
OpenAI Batch API Processor

This module provides functionality to process batch requests using the OpenAI Batch API.
It supports uploading batch files, creating batch jobs, monitoring progress, and retrieving results.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()


def setup_verbose_logging():
    """Configure verbose logging for OpenAI library to show HTTP requests."""
    # Set up logging format
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Enable debug logging for httpx (used by OpenAI client)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.DEBUG)

    # Enable debug logging for OpenAI
    openai_logger = logging.getLogger("openai")
    openai_logger.setLevel(logging.DEBUG)

    # Create a custom handler that highlights URLs
    class URLHighlightHandler(logging.StreamHandler):
        def emit(self, record):
            # Highlight URLs in the log message
            if "HTTP Request:" in record.getMessage():
                console.print(f"[bold cyan]🌐 {record.getMessage()}[/bold cyan]")
            elif "POST" in record.getMessage() or "GET" in record.getMessage():
                console.print(f"[yellow]→ {record.getMessage()}[/yellow]")
            else:
                super().emit(record)

    # Replace default handlers with our custom one for httpx
    httpx_logger.handlers = []
    httpx_logger.addHandler(URLHighlightHandler())

    console.print(
        "[green]✓[/green] Verbose logging enabled - showing all HTTP requests to OpenAI"
    )
    console.print("─" * 60)


class BatchProcessor:
    """Handles OpenAI Batch API operations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the BatchProcessor with OpenAI credentials.

        Args:
            api_key: OpenAI API key (optional, defaults to 'dummy' for proxy setups)
            base_url: OpenAI API base URL (defaults to OPENAI_BASE_URL or OPENAI_API_BASE env var)
            verbose: Enable verbose logging to show HTTP requests and URLs
        """
        # Enable verbose logging if requested
        if verbose:
            setup_verbose_logging()

        # For proxy setups, API key might not be needed
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Log the base URL being used
        if verbose:
            console.print(f"[dim]Using OpenAI base URL: {self.base_url}[/dim]")

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

        print(batch)

        # Extract errors if batch failed
        errors = None
        if batch.status == "failed" and hasattr(batch, "errors") and batch.errors:
            errors = []
            if hasattr(batch.errors, "data"):
                for error in batch.errors.data:
                    errors.append(
                        {
                            "code": getattr(error, "code", "unknown"),
                            "line": getattr(error, "line", None),
                            "message": getattr(error, "message", "Unknown error"),
                            "param": getattr(error, "param", None),
                        }
                    )

        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "failed_at": getattr(batch, "failed_at", None),
            "request_counts": request_counts,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "errors": errors,
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

        # Color status based on its value
        status_color = "green"
        if status["status"] == "failed":
            status_color = "red"
        elif status["status"] in ["in_progress", "validating", "finalizing"]:
            status_color = "yellow"

        table.add_row("ID", status["id"])
        table.add_row("Status", f"[{status_color}]{status['status']}[/{status_color}]")
        table.add_row("Created At", str(datetime.fromtimestamp(status["created_at"])))

        if status["completed_at"]:
            table.add_row(
                "Completed At", str(datetime.fromtimestamp(status["completed_at"]))
            )

        if status.get("failed_at"):
            table.add_row("Failed At", str(datetime.fromtimestamp(status["failed_at"])))

        if status["request_counts"]:
            counts = status["request_counts"]
            table.add_row("Total Requests", str(counts.get("total", 0)))
            table.add_row("Completed", str(counts.get("completed", 0)))
            table.add_row("Failed", str(counts.get("failed", 0)))

        console.print(table)

        # Display errors if batch failed
        if status["status"] == "failed" and status.get("errors"):
            console.print("\n[bold red]Batch Errors:[/bold red]")
            error_table = Table(show_header=True, header_style="bold red")
            error_table.add_column("Line", style="yellow", width=6)
            error_table.add_column("Error Code", style="cyan")
            error_table.add_column("Message", style="white")
            error_table.add_column("Parameter", style="dim")

            for error in status["errors"]:
                error_table.add_row(
                    str(error.get("line", "N/A")),
                    error.get("code", "unknown"),
                    error.get("message", "Unknown error"),
                    error.get("param", "N/A"),
                )

            console.print(error_table)

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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging to show HTTP requests and URLs",
)
def process(input_file, endpoint, window, monitor, download, output_dir, verbose):
    """Process a batch input file."""
    try:
        processor = BatchProcessor(verbose=verbose)

        # Upload file
        file_id = processor.upload_file(input_file)

        # Create batch
        batch_id = processor.create_batch(file_id, endpoint, window)

        # Monitor if requested
        if monitor:
            status = processor.monitor_batch(batch_id)

            # Handle different batch statuses
            if status["status"] == "failed":
                console.print("\n[bold red]Batch processing failed![/bold red]")
                if status.get("errors"):
                    console.print(
                        "\nPlease review the errors above and fix your batch input file."
                    )
                    console.print("Common issues:")
                    console.print("  • All requests must use the same model")
                    console.print(
                        "  • Check for invalid parameters or malformed requests"
                    )
                    console.print("  • Ensure all required fields are present")

            elif status["status"] == "completed" and download:
                # Download results if completed and requested
                output_file, error_file = processor.download_results(
                    batch_id, output_dir
                )

                if output_file:
                    # Parse and display summary
                    results = processor.parse_results(output_file)
                    console.print("\n[bold]Results Summary:[/bold]")
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

            elif status["status"] == "expired":
                console.print(
                    "\n[bold yellow]Batch expired before completion.[/bold yellow]"
                )
                console.print("The batch took too long to process and has expired.")

            elif status["status"] == "cancelled":
                console.print("\n[bold yellow]Batch was cancelled.[/bold yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--limit", default=10, help="Number of batches to list")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging to show HTTP requests and URLs",
)
def list(limit, verbose):
    """List recent batch jobs."""
    try:
        processor = BatchProcessor(verbose=verbose)
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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging to show HTTP requests and URLs",
)
def status(batch_id, verbose):
    """Get status of a specific batch job."""
    try:
        processor = BatchProcessor(verbose=verbose)
        status = processor.get_batch_status(batch_id)
        processor._display_batch_status(status)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("batch_id")
@click.option("--output-dir", default=".", help="Directory to save results")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging to show HTTP requests and URLs",
)
def download(batch_id, output_dir, verbose):
    """Download results for a completed batch."""
    try:
        processor = BatchProcessor(verbose=verbose)
        output_file, error_file = processor.download_results(batch_id, output_dir)

        if output_file:
            results = processor.parse_results(output_file)
            console.print(f"Downloaded {len(results)} results")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("batch_id")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging to show HTTP requests and URLs",
)
def cancel(batch_id, verbose):
    """Cancel a batch job."""
    try:
        processor = BatchProcessor(verbose=verbose)
        processor.cancel_batch(batch_id)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
