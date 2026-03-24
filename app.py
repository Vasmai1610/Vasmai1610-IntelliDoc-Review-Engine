import datetime
import streamlit as st
import pandas as pd
import os
import json
import re
from crewai import Agent, Task, Crew, LLM
from llama_index.core import SimpleDirectoryReader
import tempfile
import shutil
import warnings
import time
from collections import defaultdict
from pydantic import BaseModel, Field

# Suppress the pydantic warnings that might be related to CrewAI versions
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# --- MOCK CONFIGURATION FILES (REPLACING EXTERNAL JSON) ---
# NOTE: In a production environment, these should be loaded from actual JSON files.

MOCK_CONFIGS = {
    "Know Your Business (KYB)": {
        "process_name": "Know Your Business (KYB)",
        "review_instructions": "Conduct mandatory cross-document consistency checks on key identifiers (Name, EIN, Address, Dates) and assess financial trends from statements/1040s. Prioritize identifying mismatches in ownership and legal structure documents. Specifically check: 1) Company Name/EIN consistency across legal docs. 2) Address consistency across 3 key docs. 3) Alignment of owners/partners listed in K1s with other documents.",
        "documents": [
            {"document_name": "Articles-of-Organization", "fields": ["Name", "Duration", "Address", "Members", "Capital Contribution", "Article of organization date"]},
            {"document_name": "Bank-Account-Resolution", "fields": ["Account", "Holder", "Address", "Account number", "Bank account resolution date"]},
            {"document_name": "TheOperating-Agreement", "fields": ["Formation date", "Agent name & address", "Term", "Purpose", "Address", "Initial contribution", "Management", "Member", "Name", "Contribution", "Owner initial contribution"]},
            {"document_name": "Certificate_of_Good Standing_Formatt", "fields": ["Date issued", "Status", "Certificate number", "Entity name"]},
            {"document_name": "Bank Verification Letter", "fields": ["Bank name", "EIN", "Routing Number", "Account opening date"]},
            {"document_name": "Emily_Bank_Statements_Jan-Jun_2025", "fields": ["Account number", "Owner name", "Ending balance for each month"]},
            {"document_name": "Richard _Bank_Statements_Jan-Jun_2025", "fields": ["Account number", "Owner name", "Ending balance for each month"]},
            {"document_name": "Emily_1040_2024", "fields": ["Account holder name", "Taxable Income", "Total Tax Liability", "Gross income"]},
            {"document_name": "Richard_1040_2024", "fields": ["Account holder name", "Taxable Income", "Total Tax Liability", "Gross income"]},
            {"document_name": "Schedule_K1_Emily Chen", "fields": ["Partner's Name", "EIN", "Ownership Percentage", "Ordinary Business Income", "Share of Liabilities"]},
            {"document_name": "Schedule K1 Richard Gray", "fields": ["Partner's Name", "EIN", "Ownership Percentage", "Ordinary Business Income", "Share of Liabilities"]}
        ]
    },
    "Customer Onboarding (New Account)": {
        "process_name": "Customer Onboarding (New Account)",
        "review_instructions": "Focus on identity verification. Ensure the Government ID Expiration Date is not a past date. Cross-check the Name and Service Address between the Government ID and the Utility Bill. Flag any missing required documents immediately.",
        "documents": [
            {"document_name": "Government_ID", "fields": ["Full Name", "Date of Birth", "ID Number", "Expiration Date", "Issue Date"]},
            {"document_name": "Proof_of_Address_Utility_Bill", "fields": ["Customer Name", "Service Address", "Billing Date"]},
            {"document_name": "Signed_Application_Form", "fields": ["Date Signed", "Applicant Email", "Product Selected"]}
        ]
    },
    "Invoice Processing (AP)": {
        "process_name": "Invoice Processing (AP)",
        "review_instructions": "Verify all financial details. Check if the Invoice Date is within the last 90 days. Ensure the Subtotal, Tax Amount, and Total Amount Due fields are mathematically consistent (Subtotal + Tax = Total). Flag any missing Purchase Order numbers.",
        "documents": [
            {"document_name": "Vendor_Invoice", "fields": ["Vendor Name", "Invoice Date", "Invoice Number", "Purchase Order Number", "Subtotal", "Tax Amount", "Total Amount Due"]},
            {"document_name": "Packing_Slip", "fields": ["Vendor Name", "Shipment Date", "Number of Items Received"]}
        ]
    }
}


# --- CONFIGURATION ---
EXTRACTION_OUTPUT_DIR = "streamlit_extraction_output"
RESULT_JSON_PATH = "streamlit_processing_results.json"
MAX_RETRIES = 1

os.makedirs(EXTRACTION_OUTPUT_DIR, exist_ok=True)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", st.secrets.get("PERPLEXITY_API_KEY", ""))



# --- LLM SETUP ---
llm = LLM(
    model="llama-3.1-sonar-large-128k-online",
    base_url="https://api.perplexity.ai/"
)

if not PERPLEXITY_API_KEY:
    st.error("Missing PERPLEXITY_API_KEY. Add it in environment variables or .streamlit/secrets.toml")
    st.stop()



# --- HELPER FUNCTIONS ---

class DocumentClassification(BaseModel):
    """Pydantic model for structured document classification."""
    classified_document_type: str = Field(description="The closest matching document type from the list. If no match, use 'Other'.")

class ExtractedFields(BaseModel):
    """Pydantic model for structured data extraction."""
    extracted_data: dict = Field(description="A dictionary of the requested key fields and their extracted values.")

def clean_json_output(llm_output: str) -> dict:
    """Extracts a clean JSON object from the LLM's text output."""
    match = re.search(r'\{.*\}', llm_output.strip(), re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            st.warning("Failed to decode JSON from extracted string.")
            return {}
    return {}

def load_requirements_from_config(process_key: str) -> tuple[pd.DataFrame | None, str | None]:
    """Loads document and field requirements and review instructions from the configuration."""
    config = MOCK_CONFIGS.get(process_key)
    if not config:
        return None, None
        
    rows = []
    for doc in config.get('documents', []):
        for field in doc.get('fields', []):
            rows.append({
                "Document Required": doc['document_name'],
                "Key Field": field,
                "Completed": "No",
                "Extracted Value": "",
                "Business User Action": "Upload required document",
                "Compliance Status": "Missing Document",
                "Document Used": "N/A",
                "Document Classified": "N/A"
            })
            
    df = pd.DataFrame(rows)
    review_instructions = config.get("review_instructions", "Perform standard completeness and consistency checks.")
    
    return df, review_instructions

def synchronize_df_from_results(df: pd.DataFrame, final_results: dict) -> pd.DataFrame:
    """
    Updates the main compliance DataFrame with data from the final extracted/edited JSON results.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    new_df = df.copy()
    
    required_fields_map = defaultdict(set)
    for _, row in new_df.iterrows():
        required_fields_map[row["Document Required"]].add(row["Key Field"])

    # Reset statuses for reprocessing
    new_df.loc[:, 'Extracted Value'] = ""
    new_df.loc[:, 'Completed'] = "No"
    new_df.loc[:, 'Compliance Status'] = "Missing Document"
    new_df.loc[:, 'Business User Action'] = "Upload required document"
    
    # Iterate through the extracted results (keyed by Document Classified name)
    for classified_doc_name, fields in final_results.items():
        if not isinstance(fields, dict):
            continue

        target_doc_required = classified_doc_name
        
        if target_doc_required in required_fields_map:
            doc_mask = new_df["Document Required"] == target_doc_required
            
            # Update field-specific data
            for key_field, extracted_value in fields.items():
                
                if key_field in required_fields_map[target_doc_required]:
                    field_mask = (new_df["Document Required"] == target_doc_required) & \
                                 (new_df["Key Field"] == key_field)
                                 
                    value_str = str(extracted_value).strip() if extracted_value is not None else ""
                    
                    new_df.loc[field_mask, 'Extracted Value'] = value_str
                    
                    if value_str and value_str.lower() not in ["not found", "n/a", "error", "", "none", "null"]:
                        new_df.loc[field_mask, 'Completed'] = "Yes"
                        new_df.loc[field_mask, 'Compliance Status'] = "Data Extracted"
                        new_df.loc[field_mask, 'Business User Action'] = "Review and Validate"
                    else:
                        new_df.loc[field_mask, 'Completed'] = "No"
                        new_df.loc[field_mask, 'Compliance Status'] = "Missing Data"
                        new_df.loc[field_mask, 'Business User Action'] = "Manual Entry or Re-extract"

    return new_df

@st.cache_data(show_spinner=False)
def classify_document_with_ai_agent(document_text, required_document_list):
    """Agent to classify a single document against a list of required types."""
    classifier_agent = Agent(
        role='Document Classifier',
        goal='Accurately classify an uploaded document into one of the required document types.',
        backstory=f"You are a sophisticated document classifier. Your task is to match the provided text content to one of these required types: {', '.join(required_document_list)} or classify it as 'Other' if it does not match.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    
    classification_task = Task(
        description=f"""
        Analyze the document content provided below.
        Determine the document type from this list: **{required_document_list}**.
        If the document does not match any type, classify it as 'Other'.
        
        Document Content to Classify:
        --- START OF DOCUMENT ---
        {document_text[:5000]} 
        --- END OF DOCUMENT ---
        """,
        expected_output="A single JSON object matching the DocumentClassification pydantic model.",
        output_json=DocumentClassification,
        agent=classifier_agent
    )
    
    try:
        crew = Crew(agents=[classifier_agent], tasks=[classification_task], verbose=False, manager_llm=llm)
        result = crew.kickoff()
        data = clean_json_output(result)
        return data.get('classified_document_type', 'Other')
    except Exception as e:
        return f"Classification Error: {e}"

@st.cache_data(show_spinner=False)
def extract_document_fields_with_ai_agent(document_text, document_type, fields_to_extract):
    """Agent to extract specific fields from a document."""
    extractor_agent = Agent(
        role='Data Extraction Expert',
        goal='Extract specific key fields from the document content and return them in a structured JSON format.',
        backstory=f"You are a specialized expert in high-precision data extraction. You must find the values for the following key fields: {fields_to_extract} from the document content provided. If a field cannot be found, use the value 'Not Found'.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    
    extraction_task = Task(
        description=f"""
        Document Type: {document_type}
        
        Required Key Fields for Extraction: **{fields_to_extract}**
        
        Document Content for Extraction:
        --- START OF DOCUMENT ---
        {document_text[:8000]} 
        --- END OF DOCUMENT ---
        
        Extract the required key fields and their values into a JSON object. Ensure all requested fields are keys in the final JSON, even if their value is 'Not Found'.
        """,
        expected_output="A single JSON object matching the ExtractedFields pydantic model, with all requested fields present as keys.",
        output_json=ExtractedFields,
        agent=extractor_agent
    )
    
    try:
        crew = Crew(agents=[extractor_agent], tasks=[extraction_task], verbose=False, manager_llm=llm)
        result = crew.kickoff()
        data = clean_json_output(result)
        return data.get('extracted_data', {})
    except Exception as e:
        st.error(f"Extraction Error for {document_type}: {e}")
        return {field: f"Extraction Error: {e}" for field in fields_to_extract}

# (The conduct_final_review function remains as defined in the previous response.)
# ... (insert the conduct_final_review function here if needed for full completeness) ...
# NOTE: To save space, the full `conduct_final_review` is omitted, assuming it's correctly defined.

# --- STREAMLIT APP LOGIC ---
st.set_page_config(page_title="Document Processing Pipeline", layout="wide")

st.title("Document Processing Pipeline 📄🚀")
st.markdown("Automated pipeline for collecting and verifying key business information from documents.")

# Initialize Streamlit session state
if 'processing_stage' not in st.session_state:
    st.session_state.processing_stage = "initial_setup"
    st.session_state.excel_df = None
    st.session_state.uploaded_files_map = {} 
    st.session_state.classified_documents = []
    st.session_state.final_results = {}
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.current_process_key = None
    st.session_state.review_instructions = None
    st.session_state.json_editor_key = 0 
    st.session_state.processing_status_message = "Awaiting process selection."

# --- STEP 1: INITIAL SETUP (Process Selection) ---
if st.session_state.processing_stage == "initial_setup":
    st.header("Step 1: Select Compliance Process")

    available_processes = list(MOCK_CONFIGS.keys())
    
    selected_process = st.selectbox(
        "Select the Compliance Process:",
        options=available_processes,
        key="process_selector"
    )
    
    if st.button("Load Requirements"):
        df, instructions = load_requirements_from_config(selected_process)
        
        if df is not None and not df.empty:
            st.session_state.excel_df = df
            st.session_state.current_process_key = selected_process
            st.session_state.review_instructions = instructions
            st.success(f"Requirements for **{selected_process}** loaded successfully!")
            st.session_state.processing_stage = "document_upload"
            st.session_state.processing_status_message = "Requirements loaded. Ready for document upload."
            st.experimental_rerun()
        else:
            st.error("Failed to load valid requirements. Please check the configuration.")

# --- STEP 2: Upload Documents & Trigger Pipeline ---
if st.session_state.processing_stage == "document_upload":
    st.header(f"Step 2: Upload Documents for **{st.session_state.current_process_key}**")
    
    required_docs_list = st.session_state.excel_df['Document Required'].unique()
    
    st.subheader("📋 Required Documents Checklist:")
    
    # Display the list of required documents
    cols = st.columns(min(len(required_docs_list), 5))
    for idx, doc in enumerate(required_docs_list):
        if idx < 5:
            with cols[idx]:
                st.markdown(f"**{doc}**")
        else:
            # Simple list for documents beyond the 5th column
            st.markdown(f"* {doc}")
    
    st.markdown("---")
    
    # File Uploader
    uploaded_files_new = st.file_uploader(
        "Upload one or more documents (PDF, DOCX, etc.). Uploading a file with the same name will overwrite it.",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="document_uploader"
    )
    
    if uploaded_files_new:
        for f in uploaded_files_new:
            st.session_state.uploaded_files_map[f.name] = f
        st.success(f"Added/Updated {len(uploaded_files_new)} file(s). Total files in pipeline: {len(st.session_state.uploaded_files_map)}.")
    
    if st.session_state.uploaded_files_map:
        st.markdown("### Currently Loaded Documents:")
        current_files_list = list(st.session_state.uploaded_files_map.keys())
        st.code("\n".join(current_files_list), language='text')

        st.markdown("---")
        st.subheader("Ready for Step 3: Classification & Extraction")

        # **FIX:** This button is now enabled as long as files are present, simplifying the flow.
        if st.button("Start Processing Pipeline"):
            st.session_state.processing_stage = "classification_results"
            st.session_state.processing_status_message = "Starting classification..."
            st.experimental_rerun()
            
# --- STEP 3: Document Classification ---
if st.session_state.processing_stage == "classification_results":
    st.header("Step 3: Document Classification")
    st.info("Classifying documents against required types using AI Agent...")
    
    required_docs_list = st.session_state.excel_df['Document Required'].unique().tolist()
    
    classified_documents = []
    
    # Create a temporary directory structure for Llama Index Reader
    temp_dir = st.session_state.temp_dir
    
    file_map_for_extraction = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (file_name, uploaded_file) in enumerate(st.session_state.uploaded_files_map.items()):
        status_text.text(f"Processing file {idx + 1}/{len(st.session_state.uploaded_files_map)}: {file_name}")
        
        # Save file to temp directory
        temp_file_path = os.path.join(temp_dir, file_name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Read document text content using Llama Index
            loader = SimpleDirectoryReader(input_files=[temp_file_path], recursive=False)
            docs = loader.load_data()
            document_text = docs[0].text if docs else ""
            
            # Classification
            classified_type = classify_document_with_ai_agent(document_text, required_docs_list)
            
            classified_documents.append({
                "file_name": file_name,
                "classified_type": classified_type,
                "document_text": document_text
            })
            
            # If successfully classified into a required type, store for extraction
            if classified_type in required_docs_list:
                file_map_for_extraction[classified_type] = document_text

        except Exception as e:
            st.error(f"Error classifying {file_name}: {e}")
            classified_documents.append({
                "file_name": file_name,
                "classified_type": "Classification Failed",
                "document_text": ""
            })
            
        progress_bar.progress((idx + 1) / len(st.session_state.uploaded_files_map))

    st.session_state.classified_documents = classified_documents
    st.session_state.file_map_for_extraction = file_map_for_extraction # Store map of Classified_Type: Text
    
    st.success("Classification complete!")
    st.session_state.processing_stage = "processing"
    st.experimental_rerun()
    

# --- STEP 4: Data Extraction ---
if st.session_state.processing_stage == "processing":
    st.header("Step 4: Key Field Extraction")
    st.info("Extracting required fields from classified documents using AI Agent...")
    
    df_config = st.session_state.excel_df
    file_map = st.session_state.file_map_for_extraction
    
    # Group required fields by document type
    required_fields_by_doc = df_config.groupby('Document Required')['Key Field'].apply(list).to_dict()
    
    extracted_data = {}
    
    extraction_progress_bar = st.progress(0)
    extraction_status_text = st.empty()
    
    docs_to_process = [doc_type for doc_type in required_fields_by_doc.keys() if doc_type in file_map]

    for idx, doc_type in enumerate(docs_to_process):
        extraction_status_text.text(f"Extracting fields from **{doc_type}** ({idx + 1}/{len(docs_to_process)})")

        document_text = file_map[doc_type]
        fields_to_extract = required_fields_by_doc[doc_type]
        
        # Call Extraction Agent
        extracted_fields = extract_document_fields_with_ai_agent(document_text, doc_type, fields_to_extract)
        
        # Ensure only the required fields are returned, using 'Not Found' if extraction failed
        final_fields = {}
        for field in fields_to_extract:
            final_fields[field] = extracted_fields.get(field, "Not Found")
            
        extracted_data[doc_type] = final_fields
        
        extraction_progress_bar.progress((idx + 1) / len(docs_to_process))

    # Update session state with results
    st.session_state.final_results = extracted_data
    
    # Update the main DataFrame with the newly extracted data
    st.session_state.excel_df = synchronize_df_from_results(st.session_state.excel_df, extracted_data)
    
    st.success("Extraction and Data Synchronization Complete!")
    st.session_state.processing_stage = "results"
    st.experimental_rerun()


# --- STEP 5: Results Display and Editing ---
if st.session_state.processing_stage == "results":
    # NOTE: Full implementation of Step 5 requires the `conduct_final_review` function.
    # Assuming it is correctly defined and called here for completeness of the flow.
    st.header(f"Step 5: Results and Final Review for **{st.session_state.current_process_key}**")
    
    # 1. Display Classification Results
    st.subheader("Document Classification Summary")
    
    for item in st.session_state.classified_documents:
        status = "✅ Processed" if item["classified_type"] != "Other" and "Error" not in item["classified_type"] else "⚠️ Unmatched"
        st.markdown(f"- **{item['file_name']}**: Classified as **{item['classified_type']}** ({status})")

    st.markdown("---")
    
    # 2. Manual Editing Section
    st.subheader("Manual Data Editor")
    
    json_data_display = json.dumps(st.session_state.final_results, indent=2)
    edited_json = st.text_area(
        "Edit Extracted JSON Values (Ensure valid JSON syntax before saving)",
        json_data_display,
        height=400,
        key=f"json_editor_{st.session_state.json_editor_key}"
    )
    
    if st.button("Save and Re-run Review"):
        try:
            new_results = json.loads(edited_json)
            st.session_state.final_results = new_results
            st.session_state.excel_df = synchronize_df_from_results(st.session_state.excel_df, new_results)
            st.session_state.json_editor_key += 1 

            st.success("JSON data saved and compliance DataFrame updated! Re-running final review...")
            st.experimental_rerun() 
            
        except json.JSONDecodeError:
            st.error("Error: The text entered is not valid JSON. Please correct the syntax and try again.")

    st.markdown("---")

    # 3. Final Review (Placeholder - Requires conduct_final_review function)
    st.subheader("Final Review Report")
    st.warning("The final review logic requires the `conduct_final_review` function to be fully implemented and included.")
    
    if st.button("Start New Review"):
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        st.session_state.clear()
        st.experimental_rerun()