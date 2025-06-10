# Simple G AI Bot

A versatile AI bot for image analysis and dataset curation with support for multiple AI models.

## Features

- **Dataset Curation**: Generate and curate image datasets
- **Single Image Analysis**: Benchmark different AI models on individual images
- **Agentic Analysis**: Multi-step AI reasoning and analysis
- **Multiple AI Providers**: Support for OpenAI, Anthropic, and Google AI

## Quick Start

### 1. Setup Environment

```bash
cd simple_G_ai_bot
```

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Usage Examples

#### Dataset Curation
Generate 50 urban outdoor samples:
```bash
python main.py --mode data --samples 50 --urban --no-indoor
```

#### Single Image Analysis
Benchmark GPT-4o on 5 samples:
```bash
python main.py --mode benchmark --models gpt-4o --samples 5
```

#### Agentic Analysis
Run multi-step analysis with Gemini:
```bash
python main.py --mode agent --model gemini-2.5-pro --steps 10 --samples 5
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Optional |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | Optional |
| `GOOGLE_API_KEY` | Google AI API key for Gemini models | Optional |

### Command Line Options

#### Common Options
- `--mode`: Operation mode (`data`, `benchmark`, `agent`)
- `--samples`: Number of samples to process (default: 10)

#### Data Mode Options
- `--urban`: Focus on urban environments
- `--no-indoor`: Exclude indoor scenes

#### Benchmark Mode Options
- `--models`: AI model to use (e.g., `gpt-4o`, `claude-3`, `gemini-pro`)

#### Agent Mode Options
- `--model`: AI model for agentic analysis
- `--steps`: Number of reasoning steps (default: 5)

## Supported Models

- **OpenAI**: GPT-4o, GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3-opus, Claude-3-sonnet, Claude-3-haiku
- **Google**: Gemini-2.5-pro, Gemini-pro, Gemini-pro-vision

## Requirements

- Python 3.8+
- UV package manager
- Valid API keys for desired AI providers

## Installation

1. Clone the repository
2. Navigate to project directory: `cd simple_G_ai_bot`
3. Create `.env` file with your API keys
4. Install dependencies: `uv sync`
5. Run the bot with desired mode and options

## Examples

### Basic Dataset Generation
```bash
python main.py --mode data --samples 20
```

### Urban Scene Analysis
```bash
python main.py --mode data --samples 30 --urban --no-indoor
```

### Model Comparison
```bash
python main.py --mode benchmark --models gpt-4o --samples 10
python main.py --mode benchmark --models claude-3-opus --samples 10
```

### Advanced Agentic Workflow
```bash
python main.py --mode agent --model gemini-2.5-pro --steps 15 --samples 3
```

## Security Note

⚠️ **Important**: Never commit your `.env` file to version control. Add `.env` to your `.gitignore` file to keep your API keys secure.

## License

MIT License - see LICENSE file for details.