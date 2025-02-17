# PDAC Resectability Using LLMs

This repository provides a Python-based pipeline to extract structured clinical information from medical imaging reports related to pancreatic cancer. It leverages large language models (LLMs) via the Hugging Face Transformers pipeline along with custom Pydantic schemas to perform various assessments such as metastasis, lymph node status, tumor features, and vascular involvement.

## Overview

- **Multiple Assessments:** Extracts key clinical parameters including metastasis, lymph node status, tumor characteristics, and vascular resectability.
- **Large Language Model Integration:** Supports models like Llama and Gemma for text generation and structured output.
- **Configurable Workflow:** Uses a YAML configuration file (`pdac_slm_singlescript.yaml`) to define models, input data, and extraction instructions.
- **Structured Output:** Saves results into CSV files (both summary and detailed logs) for further review and analysis.

## Features

- **Metastasis Assessment:** Extracts JSON structures capturing lymph node, hematogenous, and peritoneal seeding status.
- **Lymph Node Assessment:** Parses regional and distant lymph node findings.
- **Tumor Assessment:** Extracts imaging features such as tumor location, size (in cm), morphology, duct involvement, and adjacent organ invasion.
- **Vascular Assessment:** Identifies vessel resectability status including arteries, veins, IVC, and arterial variations.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Ensure you have Python 3.8+ installed. Then run:

   ```bash
   pip install -r requirements.txt
   ```

   The key dependencies include:

   - torch
   - transformers
   - pydantic
   - datasets
   - pandas
   - pyyaml
   - tqdm

## Configuration

The pipeline is configured using the `pdac_slm_singlescript.yaml` file. This file contains:

- **models:** A list of model names (e.g., Llama, Gemma) to be used for analysis.
- **input_csv:** Path to the CSV file containing the medical reports.
- **instruct:** A list of instructions. Each instruction defines:
  - A prompt for the LLM.
  - The associated Pydantic schema for the expected output.
  - The columns to extract from the JSON response.

Customize this file as needed for your data and extraction requirements.

## Usage

Run the pipeline by executing:

```bash
python main.py
```

The script will:

1. Load the configuration and input CSV.
2. Process each report using the specified models and instructions.
3. Generate output files with structured data.

## Output

The pipeline creates a timestamped directory under `analysis_logs/` containing:

- `output.csv`: A summary CSV file with extracted fields based on the specified collection columns.
- `full_log.csv`: A detailed CSV log that includes input prompts, chain-of-thought outputs, and flattened prediction data.
- A log file (`debug.log`) is also generated to capture runtime information and any errors.

## Troubleshooting

- **Input File:** Ensure the CSV file specified in the configuration exists and the path is correct.
- **Model Availability:** Verify that the specified model names are correct and accessible. Some models may require GPU support.
- **Logs:** Consult the `debug.log` file for error messages and runtime details.

## License

This project is licensed under the Apache 2.0 License.



