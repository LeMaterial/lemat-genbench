# LeMaterial-ForgeBench

LeMaterial-ForgeBench is a tool designed to benchmark the performance of generative materials models.

## Quick Start

```bash
# Install dependencies
uv sync

# Generate reference statistics cache (optional - pre-computed cache included)
uv run scripts/compute_reference_stats.py --cache-dir ./data

# Run example
uv run examples/compute_and_use_reference_stats.py
```

## Installation

```bash
git clone https://github.com/LeMaterial/LeMaterial-ForgeBench.git
uv sync
```
or 
```bash
uv add git+https://github.com/LeMaterial/LeMaterial-ForgeBench.git
```

install orb_models using 
```bash
uv add orb_models
```

## Usage

<!-- ```bash
uv run python main.py
``` -->

