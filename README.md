# Retail Sales Forecasting Engine

[![python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A machine learning pipeline for predicting weekly retail sales volume. This project leverages a 15-week forecast horizon and hierarchical feature engineering to ensure supply chain decisions.

## Project Overview

It focuses on forecasting demand at the granular **store-product level**. By analyzing historical data (2022–2024), it predicts sales for the first 15 weeks of 2025.

## Getting Started

1. **Clone the repository**:
   `git clone https://github.com/luisgrewe/StockSalesPrediction.git`

2. **Be aware of your data the Data**:
   Locate the train and test sets in a simple `data` folder.

## Package Management with `uv`

This project uses [uv](https://docs.astral.sh/uv/) for high-performance dependency management. Below are the essential commands to get the environment running.

### Installation
If you don't have `uv` installed, run the following command for your OS:

| Platform | Command |
| :--- | :--- |
| **macOS/Linux** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Windows** | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |

---

### Environment Setup
Once `uv` is installed, navigate to the project root and run:

```bash
# Initialize the project, create .venv, and install dependencies
uv sync
```
To activate your environment, run the following comand:

| Platform | Activation Command |
| :--- | :--- |
| **macOS / Linux** | `source .venv/bin/activate` |
| **Windows (PowerShell)** | `.venv\Scripts\activate` |

---
Finally, you can run the ```salesprediction.ipynb``` file!
