!pip install langchain langchain-community chromadb sentence-transformers faiss-cpu

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Using FAISS instead of Chroma
import pickle

print("Creating vectorstore...")

documents = [
    Document(
        page_content="All personal data is encrypted using AES-256 encryption at rest and TLS 1.3 in transit. Our encryption policy covers all customer data.",
        metadata={'source': 'security_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Data retention: Customer data is retained for 7 years. After this period, data is securely deleted using DOD 5220.22-M standards.",
        metadata={'source': 'data_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Multi-factor authentication is mandatory for all employees. We use authenticator apps and hardware tokens.",
        metadata={'source': 'access_control.pdf', 'page': 1}
    ),
    Document(
        page_content="Role-based access control is implemented. Users are granted minimum necessary permissions based on their role.",
        metadata={'source': 'access_control.pdf', 'page': 2}
    ),
    Document(
        page_content="Data breach notification: In case of a security incident, affected parties will be notified within 72 hours as per GDPR requirements.",
        metadata={'source': 'incident_response.pdf', 'page': 1}
    ),
    Document(
        page_content="Our incident response plan includes: detection, containment, eradication, recovery, and post-incident review phases.",
        metadata={'source': 'incident_response.pdf', 'page': 2}
    ),
    Document(
        page_content="All employees undergo background verification before being granted access to sensitive systems and data.",
        metadata={'source': 'hr_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Annual security awareness training is mandatory for all staff. Training covers phishing, social engineering, and data protection.",
        metadata={'source': 'training_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Security audits are conducted annually by external auditors. Last audit was completed in Q4 2023.",
        metadata={'source': 'audit_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Audit logs are retained for 24 months and stored in tamper-proof systems. Logs include all access and changes.",
        metadata={'source': 'logging_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Vendor security assessments are required before onboarding. We review their security certifications and practices.",
        metadata={'source': 'vendor_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Data processing agreements are signed with all third-party processors handling personal data.",
        metadata={'source': 'dpa_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Data subjects can request access, deletion, or portability of their data via our privacy portal or email.",
        metadata={'source': 'privacy_policy.pdf', 'page': 1}
    ),
    Document(
        page_content="Our privacy notice clearly explains what data we collect, how we use it, and user rights. Notice is provided at data collection.",
        metadata={'source': 'privacy_policy.pdf', 'page': 2}
    ),
    Document(
        page_content="Vulnerability scans are performed weekly. Critical vulnerabilities are patched within 15 days, high within 30 days.",
        metadata={'source': 'security_ops.pdf', 'page': 1}
    )
]

print(f"✓ Created {len(documents)} sample documents")

# Create embeddings
print("\nCreating embeddings... (this may take 1-2 minutes)")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Create FAISS vector store (can be pickled!)
print("\nBuilding FAISS vector store...")
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

# Save vector store
with open('vectorstore.pkl', 'wb') as f:
    pickle.dump(vectorstore, f)

print(" vectorstore.pkl created successfully!")

# Verify
import os
if os.path.exists('vectorstore.pkl'):
    size = os.path.getsize('vectorstore.pkl')
    print(f" File size: {size:,} bytes")

# Test it works
print("\n Testing vectorstore...")
test_results = vectorstore.similarity_search("encryption policy", k=2)
print(f" Test successful! Found {len(test_results)} results")
