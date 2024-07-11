import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import pickle

# Load environment variables
load_dotenv()

# Set up Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Please set it in your .env file or environment variables.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Vertex AI
vertexai.init(project="core-respect-426210-t1", location="us-central1")

# Configure GCS
bucket_name = "alco-rag"

# Function to upload file to GCS
def upload_to_gcs(uploaded_file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(uploaded_file.name)
    blob.upload_from_file(uploaded_file)
    return f"gs://{bucket_name}/{uploaded_file.name}"

# Function to generate summary using Vertex AI
def generate_summary(document_uri):
    model = GenerativeModel("gemini-1.5-flash-001")
    responses = model.generate_content(
        [text1, Part.from_uri(mime_type="application/pdf", uri=document_uri), "Please summarize in detail of the document above."],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    summary = ""
    for response in responses:
        summary += response.text
    return summary

# Configurations for summary generation
text1 = """You are a very professional document summarization specialist. Given a document, your task is to strictly follow the user's instructions."""
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Function to read PDF text with metadata
def get_pdf_text(pdf_docs):
    text_chunks = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        file_name = pdf.name
        for page_number, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                text_chunks.append({"text": text, "file_name": file_name, "page_number": page_number + 1})
    return text_chunks

# Function to split text into chunks with metadata
def get_text_chunks(text_chunks):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks_with_metadata = []
    for chunk in text_chunks:
        split_texts = splitter.split_text(chunk["text"])
        for split_text in split_texts:
            chunks_with_metadata.append({
                "text": split_text,
                "file_name": chunk["file_name"],
                "page_number": chunk["page_number"]
            })
    return chunks_with_metadata

# Function to create vector store with metadata
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk["text"] for chunk in chunks]
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    metadata = {i: {"file_name": chunk["file_name"], "page_number": chunk["page_number"]} for i, chunk in enumerate(chunks)}
    vector_store.save_local("faiss_index")
    
    # Save metadata separately
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

# Function to get answer from Gemini with metadata
def get_gemini_response(question, context):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context: {context}

    Question: {question}

    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

# Function to process user input and return response with metadata
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = ""
    metadata = []
    with open("metadata.pkl", "rb") as f:
        metadata_dict = pickle.load(f)
    for i, doc in enumerate(docs):
        context += doc.page_content + "\n"
        metadata.append(metadata_dict[i])
    response = get_gemini_response(user_question, context)
    return response, metadata

# Main function
def main():
    st.set_page_config(page_title="PDF Chatbot and Summarizer", page_icon="🤖📄")

    st.title("PDF Chatbot and Summarizer")

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if st.button("Process for Chat"):
            if pdf_docs:
                with st.spinner("Processing for chat..."):
                    text_chunks = get_pdf_text(pdf_docs)
                    text_chunks_with_metadata = get_text_chunks(text_chunks)
                    get_vector_store(text_chunks_with_metadata)
                    st.success("Processing complete. You can now chat with your PDF.")
            else:
                st.warning("Please upload PDF files first.")
        
        if st.button("Generate Summary"):
            if pdf_docs:
                with st.spinner("Generating summary..."):
                    document_uri = upload_to_gcs(pdf_docs[0])  # Summarize the first uploaded PDF
                    summary = generate_summary(document_uri)
                    st.session_state.summary = summary
                    st.success("Summary generated!")
            else:
                st.warning("Please upload a PDF file first.")

    # Main content area
    tab1, tab2 = st.tabs(["Chat with PDF", "Document Summary"])

    with tab1:
        st.header("Chat with your PDF")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Chat input
        prompt = st.chat_input("Ask a question about your PDF")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, metadata = user_input(prompt)
                    response_with_metadata = response
                    # for data in metadata:
                    #     response_with_metadata += f"\n\n(Source: {data['file_name']}, Page: {data['page_number']})"
                    # st.write(response_with_metadata)
            st.session_state.messages.append({"role": "assistant", "content": response_with_metadata})

            # Rerun to update the chat display
            st.experimental_rerun()

    with tab2:
        st.header("Document Summary")
        if hasattr(st.session_state, 'summary'):
            st.write(st.session_state.summary)
        else:
            st.write("No summary generated yet. Use the 'Generate Summary' button in the sidebar to create a summary.")

if __name__ == "__main__":
    main()
