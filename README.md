# Kronos

A foundation model for financial market time series prediction. Built on top of [NeoQuasar/Kronos](https://github.com/shiyu-coder/Kronos), this fork adds Docker deployment with GPU support and a ready-to-use Web UI.

## Features

- Time series prediction for financial markets (open, high, low, close, volume)
- Three model sizes: Mini (4.1M params), Small (24.7M), Base (102.3M)
- Web-based UI for interactive predictions and visualization
- Docker deployment with NVIDIA GPU acceleration
- Models auto-downloaded from HuggingFace on first use

## Quick Start

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- A HuggingFace account (for model downloads)

### Run with Docker

1. Clone the repo:

```bash
git clone https://github.com/K227-arch/kronos.git
cd kronos
```

2. Create a `.env` file with your HuggingFace token:

```bash
HF_TOKEN=your_huggingface_token_here
```

3. Build and run:

```bash
docker compose build
docker compose up -d
```

4. Open http://localhost:7070 in your browser.

### Run Locally (without Docker)

```bash
pip install -r requirements.txt
pip install flask flask-cors plotly
python webui/app.py
```

The app runs on port 7070.

## Available Models

| Model | HuggingFace ID | Parameters | Context Length |
|-------|---------------|------------|----------------|
| Kronos-mini | `NeoQuasar/Kronos-mini` | 4.1M | 2048 |
| Kronos-small | `NeoQuasar/Kronos-small` | 24.7M | 512 |
| Kronos-base | `NeoQuasar/Kronos-base` | 102.3M | 512 |

Each model requires its corresponding tokenizer:
- Mini: `NeoQuasar/Kronos-Tokenizer-2k`
- Small/Base: `NeoQuasar/Kronos-Tokenizer-base`

## Project Structure

```
kronos/
├── model/              # Core Kronos model (tokenizer, predictor, modules)
├── webui/              # Flask web UI
│   ├── app.py          # Main application
│   └── templates/      # HTML templates
├── data/               # Place your CSV/feather data files here
├── examples/           # Usage examples and scripts
├── finetune/           # Finetuning scripts (qlib-based)
├── finetune_csv/       # Finetuning from CSV data
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose with GPU support
└── requirements.txt    # Python dependencies
```

## Data Format

Input data files (CSV or Feather) should contain these columns:

- `open`, `high`, `low`, `close` (required)
- `volume` (optional)
- `timestamp` or `date` (optional, auto-generated if missing)

Place data files in the `data/` directory for the web UI to discover them.

## License

Apache 2.0 - see [LICENSE](./LICENSE)

## Credits

Based on [Kronos](https://github.com/shiyu-coder/Kronos) by NeoQuasar.
