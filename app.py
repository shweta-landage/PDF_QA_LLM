import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile

# Load API key
load_dotenv()
groq_api_key = os.getenv("groq_QA_key")
client = Groq(api_key=groq_api_key)

# Streamlit UI
st.set_page_config(page_title="Groq PDF Q&A", page_icon="📘", layout="wide")
st.title("📘 Q&A with Your PDF using Groq LLM")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # ✅ Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name

        # ✅ Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        st.write(f"Loaded {len(documents)} pages")

        # ✅ Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # ✅ Embeddings + FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)

    st.success("PDF processed! You can now ask questions. 🎉")

    # ✅ Ask question
    query = st.text_input("Ask a question about your PDF:")

    if query:
        docs = vectorstore.similarity_search(query, k=3)

        # Combine retrieved context
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Answer the following question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        st.write("### Answer:")
        st.write(response.choices[0].message.content)




#### Here are 6 good questions you can ask your Q&A app on this document:

  #


   #Concept Clarification
#“What is a function in Python, and how do I define one?”

#Exercise Help
#“Can you explain Exercise 1.1 about calculating seconds in a day?”

#Code Examples
#“Show me an example of a for loop from the Think Python PDF.”

#Error Troubleshooting
#“Why do I get an error when I type 02492 in Python?”

#Theory Questions
#“What is the difference between while and for loops according to Think Python?”

#Practice Guidance
#“How do I solve the average pace exercise for a 10 km run?”