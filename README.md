# OpenAI Batch API Test Project

A Python project for testing and working with the OpenAI Batch API. This tool allows you to process multiple API requests efficiently using OpenAI's batch processing capabilities.

## Features

- 🚀 **Batch Processing**: Submit multiple API requests in a single batch job
- 📊 **Progress Monitoring**: Real-time monitoring of batch job status
- 💾 **Result Management**: Automatic download and parsing of results
- 🎨 **Rich CLI Interface**: Beautiful terminal output with progress indicators
- 🔧 **Flexible Configuration**: Support for environment variables and multiple endpoints
- 📝 **Request Generation**: Helper tools to create batch request files

## Setup

Simply run:

```bash
dev up
```

This command will automatically:

- Set up Python 3.11.13
- Install uv package manager
- Configure the OpenAI API endpoint for Shopify's proxy
- Install all required dependencies

## Usage

### Quick Start

1. **Process the sample batch file**:

```bash
python batch_processor.py process sample_batch_input.jsonl
```

This will:

- Upload the input file to OpenAI
- Create a batch job
- Monitor progress until completion
- Download and display results

### CLI Commands

#### Process a batch file

```bash
python batch_processor.py process <input_file> [OPTIONS]

Options:
  --endpoint TEXT          API endpoint (default: /v1/chat/completions)
  --window TEXT           Completion window (default: 24h)
  --monitor/--no-monitor  Monitor batch until completion (default: monitor)
  --download/--no-download Download results when completed (default: download)
  --output-dir TEXT       Directory to save results (default: current directory)
```

#### List recent batches

```bash
python batch_processor.py list [--limit NUMBER]
```

#### Check batch status

```bash
python batch_processor.py status <batch_id>
```

#### Download batch results

```bash
python batch_processor.py download <batch_id> [--output-dir PATH]
```

#### Cancel a batch

```bash
python batch_processor.py cancel <batch_id>
```

### Creating Batch Requests

#### Using the request generator

```bash
# Create sample chat completion requests
python create_batch_requests.py --type chat -o my_batch.jsonl

# Create embedding requests
python create_batch_requests.py --type embedding -o embeddings.jsonl

# Interactive mode - input custom prompts
python create_batch_requests.py --interactive -o custom_batch.jsonl
```

#### Manual JSONL format

Create a JSONL file where each line is a JSON object:

```json
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "How are you?"}], "max_tokens": 100}}
```

## Programmatic Usage

You can also use the `BatchProcessor` class directly in your Python code:

```python
from batch_processor import BatchProcessor
import json

# Initialize processor
processor = BatchProcessor()

# Create batch requests
requests = [
    {
        "custom_id": "req-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is Python?"}],
            "max_tokens": 100
        }
    }
]

# Save to file
with open("my_batch.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

# Process batch
file_id = processor.upload_file("my_batch.jsonl")
batch_id = processor.create_batch(file_id)

# Monitor progress
status = processor.monitor_batch(batch_id)

# Download results
if status['status'] == 'completed':
    output_file, error_file = processor.download_results(batch_id)

    # Parse results
    results = processor.parse_results(output_file)
    for result in results:
        print(f"Custom ID: {result['custom_id']}")
        print(f"Response: {result['response']['body']['choices'][0]['message']['content']}")
```

## Batch API Details

### Supported Endpoints

- `/v1/chat/completions` - Chat completions (GPT models)
- `/v1/embeddings` - Text embeddings
- `/v1/completions` - Legacy completions

### Batch Limits

- Maximum file size: 100 MB
- Maximum requests per batch: 50,000
- Completion window options: `24h` (default)
- Rate limits apply per model

### Cost Benefits

Batch API requests receive a 50% discount compared to synchronous API calls, making them ideal for:

- Large-scale data processing
- Non-time-sensitive operations
- Bulk content generation
- Dataset preparation

## Examples

### Example 1: Bulk Question Answering

```bash
# Generate a batch of questions
python create_batch_requests.py --type chat -o questions.jsonl

# Process the batch
python batch_processor.py process questions.jsonl --monitor
```

### Example 2: Generate Embeddings for a Dataset

```bash
# Create embedding requests
python create_batch_requests.py --type embedding -o embeddings.jsonl

# Process without monitoring (run in background)
python batch_processor.py process embeddings.jsonl --no-monitor

# Check status later
python batch_processor.py list
python batch_processor.py status <batch_id>
```

### Example 3: Custom Interactive Batch

```bash
# Interactive mode to create custom requests
python create_batch_requests.py --interactive -o custom.jsonl

# Process with custom completion window
python batch_processor.py process custom.jsonl --window 24h
```

## Troubleshooting

### Common Issues

1. **Connection issues**:

   - Check if `OPENAI_BASE_URL` is set correctly in `.env` or environment
   - For proxy setups, verify the proxy URL is accessible
   - For direct OpenAI access, you may need to set an API key

2. **File upload fails**:

   - Check file size (< 100 MB)
   - Ensure valid JSONL format
   - Verify each line is valid JSON

3. **Batch creation fails**:

   - Verify the endpoint is supported
   - Check request format matches endpoint requirements
   - Ensure model specified is available for batch processing

4. **Results not downloading**:
   - Wait for batch status to be 'completed'
   - Check if error file was created for failed requests

## Project Structure

```
openai-batch-test/
├── batch_processor.py       # Main batch processing module and CLI
├── create_batch_requests.py  # Helper to generate batch request files
├── sample_batch_input.jsonl  # Example batch input file
├── requirements.txt          # Python dependencies
├── .env                     # Your environment variables (optional)
└── README.md               # This file
```

## Resources

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/batch)
- [Batch API Pricing](https://openai.com/pricing)

## License

This project is provided as-is for testing and educational purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
