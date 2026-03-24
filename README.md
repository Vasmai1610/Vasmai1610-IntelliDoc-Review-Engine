# IntelliDoc Review Engine

IntelliDoc Review Engine is a professional Streamlit-based document processing application for compliance and operational workflows. It allows users to select a workflow, upload documents, classify them with AI, extract required key fields, and review results through an interactive interface.

## Repository

GitHub: `Vasmai1610/Vasmai1610-IntelliDoc-Review-Engine`

## Features

- Multi-step Streamlit workflow
- Predefined business processes:
  - Know Your Business (KYB)
  - Customer Onboarding (New Account)
  - Invoice Processing (AP)
- AI-powered document classification using CrewAI
- AI-powered field extraction with structured JSON output
- Session-based review workflow
- Manual JSON correction before final acceptance

## Tech Stack

- Python
- Streamlit
- Pandas
- CrewAI
- LlamaIndex
- Pydantic
- Perplexity-hosted LLM endpoint

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── .streamlit/
    └── secrets.toml.example

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vasmai1610/Vasmai1610-IntelliDoc-Review-Engine.git
cd Vasmai1610-IntelliDoc-Review-Engine
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure secrets

Option A: environment variable

```bash
export PERPLEXITY_API_KEY="your_api_key_here"
```

On Windows PowerShell:

```powershell
$env:PERPLEXITY_API_KEY="your_api_key_here"
```

Option B: Streamlit secrets

Create `.streamlit/secrets.toml` and add:

```toml
PERPLEXITY_API_KEY = "your_api_key_here"
```

### 5. Run the application

```bash
streamlit run app.py
```

## How It Works

1. Select a compliance or business process
2. Upload one or more supporting documents
3. Classify each document into a required document type
4. Extract configured fields from matched documents
5. Review and manually edit extracted JSON results

## Security Notes

- No API keys are hardcoded in the repository
- `.env` and `.streamlit/secrets.toml` are ignored through `.gitignore`
- Rotate any key that was previously committed or exposed

## Known Limitations

- The final review engine is referenced in the UI but not fully implemented
- If multiple uploaded files are classified into the same document type, later files may overwrite earlier ones
- Extraction quality depends on source document readability and model output quality

## Suggested Next Improvements

- Add a real final-review validation engine
- Add OCR support for scanned image-heavy files
- Support multiple files per document type
- Add test coverage and CI checks
- Move mock configurations into external JSON or YAML files

## License

This project is licensed under the MIT License.
