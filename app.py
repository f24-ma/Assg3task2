import streamlit as st
import time
import json
import os
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

def wait_for_rate_limit(min_delay=3):
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

def analyze_document(pdf_bytes, pdf_name, rules, max_retries=3):
    """Analyze document with retry logic and exponential backoff"""
    
    for attempt in range(max_retries):
        tmp_path = None
        uploaded_file = None
        
        try:
            # Wait for rate limit
            wait_for_rate_limit(min_delay=3)
            
            # Configure model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Create prompt
            rules_text = "\n".join([f"{i+1}. {rule['name']}: {rule['description']}" for i, rule in enumerate(rules)])
            
            prompt = f"""Analyze this PDF document for compliance with these {len(rules)} rules:

{rules_text}

For EACH rule above, provide:
- Status: PASS, FAIL, or WARNING
- Details: Brief explanation (1-2 sentences)

Return ONLY valid JSON in this format (no markdown, no code blocks):
{{
  "findings": [
    {{"rule": "Data Privacy", "status": "PASS", "details": "Explanation here"}},
    {{"rule": "Financial Reporting", "status": "WARNING", "details": "Explanation here"}}
  ]
}}"""
            
            # Save PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            # Upload PDF to Gemini
            uploaded_file = genai.upload_file(path=tmp_path, mime_type='application/pdf')
            
            # Wait for file to be processed
            time.sleep(2)
            
            # Generate content
            response = model.generate_content([prompt, uploaded_file])
            
            # Parse response
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            
            # Validate response
            if 'findings' in result and len(result['findings']) > 0:
                # Clean up
                if uploaded_file:
                    uploaded_file.delete()
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
                return result
            else:
                raise ValueError("No findings in response")
        
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to parse API response after {max_retries} attempts")
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
                time.sleep(2)
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle quota/rate limit errors
            if "429" in str(e) or "quota" in error_msg or "rate" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5
                    st.warning(f"Rate limit hit. Waiting {wait_time} seconds... (retry {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("API Quota Exceeded")
                    st.info("Solutions:")
                    st.info("1. Wait a few minutes and try again")
                    st.info("2. Check your quota: https://ai.dev/usage")
                    st.info("3. Get a new API key: https://aistudio.google.com/apikey")
                    return None
            
            # Handle invalid API key
            elif "api key" in error_msg or "invalid" in error_msg:
                st.error("Invalid API Key")
                st.info("Please check your API key at: https://aistudio.google.com/apikey")
                return None
            
            # Handle other errors
            else:
                if attempt < max_retries - 1:
                    st.warning(f"Error on attempt {attempt + 1}: {str(e)}")
                    time.sleep(2)
                    continue
                else:
                    st.error(f"Analysis failed: {str(e)}")
                    return None
        
        finally:
            # Always clean up resources
            try:
                if uploaded_file:
                    uploaded_file.delete()
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
        help="Get your API key from https://aistudio.google.com/apikey"
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
        st.metric("Compliance Rate", "--")

with col4:
    if st.session_state.results:
        issues = st.session_state.results.get('failed', 0)
        st.metric("Issues", issues)
    else:
        st.metric("Issues", "--")

st.markdown("---")

# Run Audit Button - Centered and prominent
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    audit_button = st.button(
        "Run Compliance Audit", 
        type="primary", 
        use_container_width=True,
        disabled=st.session_state.processing
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
        st.metric("Passed", results['passed'])
    
    with col2:
        st.metric("Warnings", results['warnings'])
    
    with col3:
        st.metric("Failed", results['failed'])
    
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
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        results_json = json.dumps(results, indent=2)
        st.download_button(
            label="Export Results as JSON",
            data=results_json,
            file_name=f"compliance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

else:
    # Empty state
    st.info("Upload documents and click 'Run Compliance Audit' to get started")

# Footer
st.markdown("---")
st.caption(f"Compliance Checker v1.0 | Powered by Google Gemini | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
