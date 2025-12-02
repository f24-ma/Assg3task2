import streamlit as st
import time
import json
from datetime import datetime
import google.generativeai as genai
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Policy Compliance Checker",
    layout="wide"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0

# Compliance rules
COMPLIANCE_RULES = [
    {'name': 'Data Privacy', 'description': 'GDPR and data protection compliance'},
    {'name': 'Financial Reporting', 'description': 'SOX and financial disclosure requirements'},
    {'name': 'Anti-Corruption', 'description': 'FCPA and anti-bribery provisions'},
    {'name': 'Employment Law', 'description': 'Labor regulations and workplace standards'},
    {'name': 'Environmental', 'description': 'EPA and environmental regulations'},
    {'name': 'Intellectual Property', 'description': 'Patent, trademark, and copyright compliance'},
    {'name': 'Health & Safety', 'description': 'OSHA and workplace safety requirements'},
    {'name': 'Export Control', 'description': 'ITAR and export compliance'},
    {'name': 'Consumer Protection', 'description': 'FTC and consumer rights regulations'},
    {'name': 'Securities Law', 'description': 'SEC disclosure and trading regulations'},
    {'name': 'Antitrust', 'description': 'Sherman Act and competition law'},
    {'name': 'Tax Compliance', 'description': 'IRS and tax reporting requirements'},
    {'name': 'Healthcare', 'description': 'HIPAA and healthcare regulations'},
    {'name': 'Telecommunications', 'description': 'FCC and telecom regulations'},
    {'name': 'Insurance', 'description': 'State insurance commission requirements'}
]

def wait_for_rate_limit(min_delay=2):
    """Implement rate limiting to avoid 429 errors"""
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_request_time
    
    if time_since_last < min_delay:
        wait_time = min_delay - time_since_last
        time.sleep(wait_time)
    
    st.session_state.last_request_time = time.time()

def configure_api(api_key):
    """Configure Google Generative AI API with error handling"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"API Configuration Error: {str(e)}")
        return False

def analyze_document_with_retry(pdf_bytes, pdf_name, rules, max_retries=3):
    """Analyze document with retry logic and exponential backoff"""
    
    for attempt in range(max_retries):
        try:
            # Wait for rate limit
            wait_for_rate_limit(min_delay=3)
            
            # Configure model with lower settings to reduce quota usage
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Create prompt
            rules_text = "\n".join([f"{i+1}. {rule['name']}: {rule['description']}" for i, rule in enumerate(rules)])
            
            prompt = f"""Analyze this PDF document for compliance with the following rules:

{rules_text}

IMPORTANT: You must provide a finding for EACH rule listed above.

For each rule, determine:
- Status: PASS (compliant), FAIL (non-compliant), or WARNING (needs review)
- Details: Brief explanation of your finding (1-2 sentences)

Return your analysis in this EXACT JSON format:
{{
  "findings": [
    {{
      "rule": "Data Privacy",
      "status": "PASS",
      "details": "Your explanation here"
    }}
  ]
}}

Analyze the document now and provide findings for all {len(rules)} rules."""
            
            # Save PDF to temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Upload the PDF file
                uploaded_file = genai.upload_file(path=tmp_path, mime_type='application/pdf', display_name=pdf_name)
                
                # Generate content with the uploaded file
                response = model.generate_content([prompt, uploaded_file])
                
                # Delete the uploaded file to save quota
                uploaded_file.delete()
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # Parse response
            try:
                # Extract JSON from response
                text = response.text.strip()
                
                # Try to find JSON in various formats
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(text)
                
                # Validate that we have findings
                if 'findings' in result and len(result['findings']) > 0:
                    return result
                else:
                    raise ValueError("No findings in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                st.warning(f"JSON parsing issue on attempt {attempt + 1}: {str(e)}")
                # If JSON parsing fails, try to extract meaningful info from text
                if attempt == max_retries - 1:
                    # Last attempt - return default findings
                    return {
                        "findings": [
                            {
                                "rule": rule['name'],
                                "status": "WARNING",
                                "details": "Analysis completed but results format was unclear. Manual review recommended."
                            } for rule in rules
                        ]
                    }
                continue
        
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific errors
            if "429" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 5
                    st.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("API Quota Exceeded")
                    st.info("Solutions:")
                    st.info("1. Wait a few minutes and try again")
                    st.info("2. Check your quota at: https://ai.dev/usage")
                    st.info("3. Get a new API key at: https://aistudio.google.com/apikey")
                    st.info("4. Consider upgrading your plan for higher limits")
                    return None
            elif "api key not valid" in error_msg.lower() or "invalid api key" in error_msg.lower():
                st.error("Invalid API Key")
                st.info("Please check your API key and try again. Get a valid key at: https://aistudio.google.com/apikey")
                return None
            else:
                st.error(f"Error analyzing document (attempt {attempt + 1}/{max_retries}): {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
    
    return None

# Header
st.title("Policy Compliance Checker")
st.markdown("Automated document compliance analysis powered by Google Gemini")

# Sidebar - Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key Input
    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        value=st.session_state.api_key,
        help="Get your API key from https://aistudio.google.com/apikey"
    )
    
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        if api_key_input:
            if configure_api(api_key_input):
                st.success("API Key configured")
    
    st.markdown("---")
    
    # Rule Selection
    st.subheader("Compliance Rules")
    selected_rules = []
    
    with st.expander("Select Rules to Check", expanded=True):
        # Select all checkbox
        select_all = st.checkbox("Select All", value=True)
        
        for rule in COMPLIANCE_RULES:
            if select_all:
                checked = st.checkbox(
                    rule['name'],
                    value=True,
                    key=rule['name'],
                    help=rule['description']
                )
            else:
                checked = st.checkbox(
                    rule['name'],
                    value=False,
                    key=rule['name'],
                    help=rule['description']
                )
            
            if checked:
                selected_rules.append(rule)
    
    st.markdown("---")
    
    # Rate Limit Info
    st.info("""
    **Rate Limit Tips:**
    - Free tier: ~15 requests/min
    - App waits 3 seconds between requests
    - Uses Gemini Flash for lower quota usage
    """)

# Main Content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rules", len(selected_rules))

with col2:
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Maximum 200MB per file"
    )
    docs_count = len(uploaded_files) if uploaded_files else 0
    st.metric("Documents", docs_count)

with col3:
    if st.session_state.results:
        compliance_rate = st.session_state.results.get('compliance_rate', 0)
        st.metric("Compliance Rate", f"{compliance_rate}%")
    else:
        st.metric("Compliance Rate", "--")

with col4:
    if st.session_state.results:
        issues = st.session_state.results.get('failed', 0)
        st.metric("Issues", issues)
    else:
        st.metric("Issues", "--")

st.markdown("---")

# Run Audit Button
if st.button("Run Compliance Audit", type="primary", use_container_width=True):
    
    # Validation
    if not st.session_state.api_key:
        st.error("Please enter your Google API Key in the sidebar")
        st.info("Get your free API key at: https://aistudio.google.com/apikey")
    elif not uploaded_files:
        st.error("Please upload at least one PDF document")
    elif not selected_rules:
        st.error("Please select at least one compliance rule")
    else:
        # Configure API
        if not configure_api(st.session_state.api_key):
            st.stop()
        
        # Process documents
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        total_passed = 0
        total_failed = 0
        total_warnings = 0
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
            
            try:
                # Read PDF
                pdf_bytes = uploaded_file.read()
                
                # Analyze document
                result = analyze_document_with_retry(
                    pdf_bytes,
                    uploaded_file.name,
                    selected_rules
                )
                
                if result and 'findings' in result:
                    # Count statuses
                    for finding in result['findings']:
                        if finding['status'] == 'PASS':
                            total_passed += 1
                        elif finding['status'] == 'FAIL':
                            total_failed += 1
                        else:
                            total_warnings += 1
                    
                    all_results.append({
                        'document_name': uploaded_file.name,
                        'findings': result['findings']
                    })
                else:
                    st.error(f"Failed to analyze {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("Analysis complete!")
        
        # Calculate compliance rate
        total_checks = total_passed + total_failed + total_warnings
        compliance_rate = int((total_passed / total_checks * 100)) if total_checks > 0 else 0
        
        # Store results
        st.session_state.results = {
            'compliance_rate': compliance_rate,
            'passed': total_passed,
            'failed': total_failed,
            'warnings': total_warnings,
            'document_results': all_results,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.success(f"Analysis complete! Compliance Rate: {compliance_rate}%")
        st.rerun()

# Display Results
if st.session_state.results:
    st.markdown("---")
    st.header("Results")
    
    results = st.session_state.results
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Passed", results['passed'], delta=None)
    
    with col2:
        st.metric("Warnings", results['warnings'], delta=None)
    
    with col3:
        st.metric("Failed", results['failed'], delta=None)
    
    # Compliance rate bar
    st.subheader("Overall Compliance")
    st.progress(results['compliance_rate'] / 100)
    
    # Document results
    st.subheader("Document Details")
    
    for doc_result in results['document_results']:
        with st.expander(f"{doc_result['document_name']}", expanded=True):
            for finding in doc_result['findings']:
                status = finding['status']
                
                if status == 'PASS':
                    st.success(f"**{finding['rule']}**: {finding['details']}")
                elif status == 'FAIL':
                    st.error(f"**{finding['rule']}**: {finding['details']}")
                else:
                    st.warning(f"**{finding['rule']}**: {finding['details']}")
    
    # Export results
    st.markdown("---")
    if st.button("Export Results as JSON"):
        results_json = json.dumps(results, indent=2)
        st.download_button(
            label="Download JSON",
            data=results_json,
            file_name=f"compliance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

else:
    # Empty state
    st.info("Upload documents and click 'Run Compliance Audit' to get started")

# Footer
st.markdown("---")
st.caption("Compliance Checker v1.0 | Powered by Google Gemini")
