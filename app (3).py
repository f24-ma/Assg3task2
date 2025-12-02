import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import pypdf
import tempfile

st.set_page_config(page_title="Compliance Checker", layout="wide")

if "results_df" not in st.session_state:
    st.session_state.results_df = None

st.title("Policy Compliance Checker")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google API Key", type="password")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    run_audit = st.button("Run Audit", type="primary", use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rules", "15")
col2.metric("Documents", len(uploaded_files) if uploaded_files else 0)

if st.session_state.results_df is not None:
    df = st.session_state.results_df
    compliant = len(df[df["Status"] == "COMPLIANT"])
    col3.metric("Compliance Rate", f"{compliant/len(df)*100:.1f}%")
    col4.metric("Issues", len(df[df["Status"] == "NON_COMPLIANT"]))
else:
    col3.metric("Compliance Rate", "--")
    col4.metric("Issues", "--")

st.markdown("---")

tab1, tab2 = st.tabs(["Results", "Analytics"])

with tab1:
    if st.session_state.results_df is not None:
        st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
        csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Report", csv, f"report_{datetime.now().strftime('%Y%m%d')}.csv")
    else:
        st.info("Upload documents and run audit to see results")

with tab2:
    if st.session_state.results_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            category_data = st.session_state.results_df.groupby(["Category", "Status"]).size().reset_index(name="Count")
            fig = px.bar(category_data, x="Category", y="Count", color="Status", title="By Category")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            severity_data = st.session_state.results_df.groupby(["Severity", "Status"]).size().reset_index(name="Count")
            fig = px.bar(severity_data, x="Severity", y="Count", color="Status", title="By Severity")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run audit to see analytics")

if run_audit:
    if not api_key:
        st.error("Enter API key")
    elif not uploaded_files:
        st.error("Upload PDFs")
    else:
        with st.spinner("Processing..."):
            try:
                os.environ["GOOGLE_API_KEY"] = api_key
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    documents = []
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        with open(temp_path, "rb") as file:
                            pdf_reader = pypdf.PdfReader(file)
                            for page_num, page in enumerate(pdf_reader.pages):
                                text = page.extract_text()
                                if text.strip():
                                    documents.append(Document(
                                        page_content=text,
                                        metadata={"source": uploaded_file.name, "page": page_num + 1}
                                    ))
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=temp_dir)
                    
                    rules = [
                        {"id": "RULE_001", "category": "Data Protection", "name": "Personal Data Encryption", "description": "encrypted data rest transit", "severity": "CRITICAL"},
                        {"id": "RULE_002", "category": "Data Protection", "name": "Data Retention Policy", "description": "retention periods years", "severity": "HIGH"},
                        {"id": "RULE_003", "category": "Access Control", "name": "Multi-Factor Authentication", "description": "MFA mandatory users", "severity": "CRITICAL"},
                        {"id": "RULE_004", "category": "Access Control", "name": "Role-Based Access Control", "description": "RBAC least privilege", "severity": "HIGH"},
                        {"id": "RULE_005", "category": "Incident Response", "name": "Breach Notification", "description": "breaches reported 72 hours", "severity": "CRITICAL"},
                        {"id": "RULE_006", "category": "Incident Response", "name": "Response Plan", "description": "incident response procedures", "severity": "HIGH"},
                        {"id": "RULE_007", "category": "Employee Management", "name": "Background Verification", "description": "background checks employees", "severity": "MEDIUM"},
                        {"id": "RULE_008", "category": "Employee Management", "name": "Security Training", "description": "annual training employees", "severity": "MEDIUM"},
                        {"id": "RULE_009", "category": "Audit and Compliance", "name": "Security Audits", "description": "audits annually", "severity": "HIGH"},
                        {"id": "RULE_010", "category": "Audit and Compliance", "name": "Log Retention", "description": "logs retained 12 months", "severity": "HIGH"},
                        {"id": "RULE_011", "category": "Third-Party", "name": "Vendor Assessment", "description": "vendors security assessment", "severity": "HIGH"},
                        {"id": "RULE_012", "category": "Third-Party", "name": "Data Processing Agreements", "description": "DPAs third parties", "severity": "CRITICAL"},
                        {"id": "RULE_013", "category": "Privacy Rights", "name": "Data Subject Rights", "description": "data access deletion 30 days", "severity": "CRITICAL"},
                        {"id": "RULE_014", "category": "Privacy Rights", "name": "Privacy Notice", "description": "privacy notice collection", "severity": "HIGH"},
                        {"id": "RULE_015", "category": "System Security", "name": "Vulnerability Management", "description": "vulnerabilities patched 30 days", "severity": "CRITICAL"}
                    ]
                    
                    results = []
                    progress = st.progress(0)
                    
                    for i, rule in enumerate(rules):
                        docs = vectorstore.similarity_search(rule["description"], k=5)
                        keywords = rule["description"].split()
                        evidence = sum(1 for doc in docs if any(kw in doc.page_content.lower() for kw in keywords))
                        
                        status = "COMPLIANT" if evidence >= 3 else "PARTIAL" if evidence >= 1 else "NON_COMPLIANT"
                        confidence = 0.8 if evidence >= 3 else 0.5 if evidence >= 1 else 0.3
                        
                        results.append({
                            "Rule ID": rule["id"],
                            "Rule Name": rule["name"],
                            "Category": rule["category"],
                            "Severity": rule["severity"],
                            "Status": status,
                            "Confidence": f"{confidence:.0%}",
                            "Evidence": evidence
                        })
                        progress.progress((i + 1) / len(rules))
                    
                    st.session_state.results_df = pd.DataFrame(results)
                    progress.empty()
                    st.success("Audit complete!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Compliance Checker v1.0</div>", unsafe_allow_html=True)
