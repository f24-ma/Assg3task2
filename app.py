# Disable ChromaDB telemetry FIRST (before any other imports)
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import streamlit as st
import time
import json
import tempfile
from datetime import datetime

# Only import if API key is provided
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.error("google-generativeai not installed. Run: pip install google-generativeai")

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
if 'processing' not in st.session_state:
    st.session_state.processing = False

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

def wait_for_rate_limit(min_delay=5):
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

def analyze_document(pdf_bytes, pdf_name, rules, max_retries=2):
    """Analyze document with retry logic and exponential backoff"""
    
    for attempt in range(max_retries):
        tmp_path = None
        uploaded_file = None
        
        try:
            # Wait for rate limit (increased to 5 seconds)
            wait_for_rate_limit(min_delay=5)
            
            # Configure model with lower temperature for more consistent JSON
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
            
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=generation_config
            )
            
            # Create prompt with strict JSON formatting instructions
            rules_text = "\n".join([f"{i+1}. {rule['name']}: {rule['description']}" for i, rule in enumerate(rules)])
            
            prompt = f"""You are a compliance analyzer. Analyze this PDF document for compliance with these {len(rules)} rules:

{rules_text}

CRITICAL: You must respond with ONLY a valid JSON object. No explanations, no markdown, no code blocks.

For EACH rule above, provide a finding with:
- rule: exact rule name from the list
- status: must be exactly "PASS", "FAIL", or "WARNING"
- details: brief explanation (1-2 sentences)

Example format:
{{"findings": [{{"rule": "Data Privacy", "status": "PASS", "details": "Document complies with GDPR requirements."}}]}}

Now analyze the document and return ONLY the JSON object:"""
            
            # Save PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            # Upload PDF to Gemini with retry
            max_upload_retries = 3
            uploaded_file = None
            
            for upload_attempt in range(max_upload_retries):
                try:
                    uploaded_file = genai.upload_file(path=tmp_path, mime_type='application/pdf')
                    break
                except Exception as upload_error:
                    if upload_attempt < max_upload_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise upload_error
            
            if not uploaded_file:
                raise Exception("Failed to upload file after retries")
            
            # Wait for file to be processed with timeout
            max_wait = 30  # 30 seconds timeout
            start_wait = time.time()
            
            while (time.time() - start_wait) < max_wait:
                time.sleep(2)
                try:
                    file_status = genai.get_file(uploaded_file.name)
                    if file_status.state.name == "ACTIVE":
                        break
                except:
                    continue
            
            # Final check
            file_status = genai.get_file(uploaded_file.name)
            if file_status.state.name != "ACTIVE":
                raise Exception(f"File not ready after {max_wait} seconds")
            
            # Generate content
            response = model.generate_content([prompt, uploaded_file])
            
            # Parse response
            text = response.text.strip()
            
            # Clean up response - remove any markdown formatting
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing text that isn't JSON
            if text.startswith('{'):
                text = '{' + text.split('{', 1)[1]
            if text.endswith('}'):
                text = text.rsplit('}', 1)[0] + '}'
            
            result = json.loads(text)
            
            # Validate response structure
            if 'findings' not in result:
                raise ValueError("Response missing 'findings' key")
            
            if not isinstance(result['findings'], list):
                raise ValueError("'findings' must be a list")
            
            if len(result['findings']) == 0:
                raise ValueError("No findings in response")
            
            # Validate each finding
            for finding in result['findings']:
                if 'rule' not in finding or 'status' not in finding or 'details' not in finding:
                    raise ValueError("Finding missing required fields")
                if finding['status'] not in ['PASS', 'FAIL', 'WARNING']:
                    finding['status'] = 'WARNING'  # Fix invalid status
            
            # Clean up
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
            
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            return result
        
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to parse API response after {max_retries} attempts")
                st.error(f"Error: {str(e)}")
                # Return fallback results
                return {
                    "findings": [
                        {
                            "rule": rule['name'],
                            "status": "WARNING",
                            "details": "Analysis completed but response format was unclear. Manual review recommended."
                        } for rule in rules
                    ]
                }
            else:
                st.warning(f"JSON parsing error on attempt {attempt + 1}, retrying...")
                time.sleep(3)
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle quota/rate limit errors
            if "429" in str(e) or "quota" in error_msg or "rate limit" in error_msg or "resource exhausted" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10  # Increased backoff
                    st.warning(f"Rate limit hit. Waiting {wait_time} seconds... (retry {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("API Quota Exceeded")
                    st.info("**Solutions:**")
                    st.info("1. Wait 1 minute and try again")
                    st.info("2. Check your quota: https://aistudio.google.com/app/apikey")
                    st.info("3. Get a new API key: https://aistudio.google.com/apikey")
                    return None
            
            # Handle invalid API key
            elif "api key" in error_msg or "invalid" in error_msg or "authentication" in error_msg:
                st.error("Invalid API Key")
                st.info("Get your API key at: https://aistudio.google.com/apikey")
                return None
            
            # Handle file upload errors
            elif "file" in error_msg and ("upload" in error_msg or "processing" in error_msg):
                if attempt < max_retries - 1:
                    st.warning(f"File processing error on attempt {attempt + 1}, retrying in 10 seconds...")
                    time.sleep(10)
                    continue
                else:
                    st.error(f"File upload failed after {max_retries} attempts: {str(e)}")
                    st.info("Try these solutions:")
                    st.info("1. Ensure PDF is not corrupted")
                    st.info("2. Try a smaller file size")
                    st.info("3. Wait a minute and try again")
                    return None
            
            # Handle other errors
            else:
                if attempt < max_retries - 1:
                    st.warning(f"Error on attempt {attempt + 1}: {str(e)}")
                    time.sleep(5)
                    continue
                else:
                    st.error(f"Analysis failed: {str(e)}")
                    return None
        
        finally:
            # Always clean up resources
            try:
                if uploaded_file:
                    genai.delete_file(uploaded_file.name)
            except:
                pass
            
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
    
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
        help="Get your API key from https://aistudio.google.com/apikey",
        placeholder="Enter your API key here"
    )
    
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        if api_key_input and GENAI_AVAILABLE:
            if configure_api(api_key_input):
                st.success("API Key configured")
    
    st.markdown("---")
    
    # Rule Selection
    st.subheader("Compliance Rules")
    
    # Select all checkbox
    select_all = st.checkbox("Select All Rules", value=True)
    
    selected_rules = []
    
    with st.expander("View/Edit Rules", expanded=False):
        for rule in COMPLIANCE_RULES:
            checked = st.checkbox(
                rule['name'],
                value=select_all,
                key=f"rule_{rule['name']}",
                help=rule['description']
            )
            
            if checked:
                selected_rules.append(rule)
    
    if select_all:
        selected_rules = COMPLIANCE_RULES.copy()
    
    st.markdown("---")
    
    # Rate Limit Info
    st.info("""
**Rate Limit Tips:**
- Free tier: 15 requests/min
- App waits 5 seconds between requests
- Uses Gemini Flash 1.5 for efficiency
- For large docs, analysis takes 3-7 minutes
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
        help="Maximum 200MB per file",
        key="pdf_uploader"
    )
    docs_count = len(uploaded_files) if uploaded_files else 0
    st.metric("Documents", docs_count)

with col3:
    if st.session_state.results:
        compliance_rate = st.session_state.results.get('compliance_rate', 0)
        st.metric("Compliance Rate", f"{compliance_rate}%")
    else:
        st.metric("Compliance Rate", "0%")

with col4:
    if st.session_state.results:
        issues = st.session_state.results.get('failed', 0)
        st.metric("Issues", issues)
    else:
        st.metric("Issues", "0")

st.markdown("---")

# Status Check
api_configured = bool(st.session_state.api_key)
docs_uploaded = bool(uploaded_files)
rules_selected = bool(selected_rules)

col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    if api_configured:
        st.success("API Key Configured")
    else:
        st.warning("API Key Missing")

with col_status2:
    if docs_uploaded:
        st.success(f"{docs_count} Document(s) Uploaded")
    else:
        st.warning("No Documents Uploaded")

with col_status3:
    if rules_selected:
        st.success(f"{len(selected_rules)} Rules Selected")
    else:
        st.warning("No Rules Selected")

st.markdown("---")

# Run Audit Button - Large and prominent
st.markdown("### Ready to Analyze")

if st.session_state.processing:
    st.warning("Analysis in progress... Please wait.")
    audit_button = False
else:
    audit_button = st.button(
        "Run Compliance Audit", 
        type="primary", 
        use_container_width=True,
        help="Click to start analyzing your documents",
        disabled=not (api_configured and docs_uploaded and rules_selected)
    )

st.markdown("---")

# Process audit
if audit_button:
    
    # Validation
    if not GENAI_AVAILABLE:
        st.error("Google Generative AI library not available")
        st.stop()
    
    if not st.session_state.api_key:
        st.error("Please enter your Google API Key in the sidebar")
        st.info("Get your free API key at: https://aistudio.google.com/apikey")
        st.stop()
    
    if not uploaded_files:
        st.error("Please upload at least one PDF document")
        st.stop()
    
    if not selected_rules:
        st.error("Please select at least one compliance rule")
        st.stop()
    
    # Configure API
    if not configure_api(st.session_state.api_key):
        st.stop()
    
    # Set processing flag
    st.session_state.processing = True
    
    # Process documents
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    total_passed = 0
    total_failed = 0
    total_warnings = 0
    
    try:
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
            
            try:
                # Read PDF
                pdf_bytes = uploaded_file.read()
                
                # Analyze document
                result = analyze_document(
                    pdf_bytes,
                    uploaded_file.name,
                    selected_rules
                )
                
                if result and 'findings' in result:
                    # Count statuses
                    for finding in result['findings']:
                        status = finding.get('status', 'WARNING').upper()
                        if status == 'PASS':
                            total_passed += 1
                        elif status == 'FAIL':
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
        
    finally:
        st.session_state.processing = False
        time.sleep(1)
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
    st.write(f"**{results['compliance_rate']}%** compliant")
    
    # Document results
    st.subheader("Document Details")
    
    for doc_result in results['document_results']:
        with st.expander(f"{doc_result['document_name']}", expanded=True):
            for finding in doc_result['findings']:
                status = finding.get('status', 'WARNING').upper()
                rule = finding.get('rule', 'Unknown Rule')
                details = finding.get('details', 'No details provided')
                
                if status == 'PASS':
                    st.success(f"**{rule}**: {details}")
                elif status == 'FAIL':
                    st.error(f"**{rule}**: {details}")
                else:
                    st.warning(f"**{rule}**: {details}")
    
    # Export results
    st.markdown("---")
    
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="Export Results as JSON",
        data=results_json,
        file_name=f"compliance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

else:
    # Empty state with instructions
    st.markdown("---")
    st.info("Getting Started")
    st.markdown("""
    **Follow these steps to analyze your documents:**
    
    1. **Enter API Key** - Add your Google API key in the left sidebar
       - Get one free at: https://aistudio.google.com/apikey
    
    2. **Upload PDF** - Click "Browse files" above to upload your document(s)
    
    3. **Select Rules** - Choose which compliance rules to check (15 available)
    
    4. **Run Analysis** - Click "Run Compliance Audit" button above
    
    **Note:** Analysis typically takes 2-5 minutes depending on document size.
    """)

# Footer
st.markdown("---")
st.caption(f"Compliance Checker v1.0 | Powered by Google Gemini 1.5 Flash | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
