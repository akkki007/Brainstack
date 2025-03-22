import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
import tempfile
from dotenv import load_dotenv
from yt_dlp import YoutubeDL  # For downloading YouTube videos
from pathlib import Path

# Load environment variables
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

# Set page title
st.title("LangChain RAG with Groq and Video Summarizer")

# Initialize embeddings if not already done
if "embeddings" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Create sidebar for source selection
st.sidebar.title("Configure Source")
source_type = st.sidebar.radio("Select Source Type", ["Web URL", "PDF Upload", "Video Summarizer"])

# Function to process web content
def process_web_content(url):
    with st.spinner("Loading and processing web content..."):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            # Show document info
            st.sidebar.success(f"Loaded {len(docs)} documents from web")
            
            # Process documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs[:50])  # Limit to first 50 docs
            
            # Create vector store
            vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
            
            st.sidebar.success(f"Created {len(final_documents)} chunks")
            
            return vectors
        except Exception as e:
            st.sidebar.error(f"Error processing web content: {str(e)}")
            return None

# Function to process PDF content
def process_pdf(pdf_file):
    with st.spinner("Processing PDF..."):
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # Write the uploaded file to the temporary file
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load the PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Show document info
            st.sidebar.success(f"Loaded {len(docs)} pages from PDF")
            
            # Check for empty content
            if not docs:
                st.sidebar.error("No content found in the PDF. Please upload a valid PDF with extractable text.")
                os.unlink(tmp_path)
                return None
                
            # Process documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            
            # Show chunk info
            chunk_count = len(final_documents)
            st.sidebar.success(f"Created {chunk_count} chunks")
            
            # Handle the case when no chunks are created
            if chunk_count == 0:
                st.sidebar.warning("No chunks created, trying with smaller chunk size")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                final_documents = text_splitter.split_documents(docs)
                chunk_count = len(final_documents)
                st.sidebar.info(f"Second attempt: Created {chunk_count} chunks")
                
                # If still no chunks, use original documents
                if chunk_count == 0:
                    st.sidebar.warning("Using original documents as chunks")
                    final_documents = docs
            
            # Create vector store
            vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
            
            return vectors
        except Exception as e:
            st.sidebar.error(f"Error processing PDF: {str(e)}")
            return None
        finally:
            # Clean up temporary file
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

# Function to download YouTube video using yt-dlp
def download_youtube_video(url, output_path="."):
    try:
        # Specify the FFmpeg installation path
        ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # Replace with your FFmpeg path
        # ffmpeg_path = "/usr/local/bin/ffmpeg"  # For macOS/Linux

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f"{output_path}/%(title)s.%(ext)s",
            'quiet': True,
            'merge_output_format': 'mp4',  # Ensure the output is in MP4 format
            'ffmpeg_location': ffmpeg_path,  # Specify the FFmpeg path
        }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info_dict)
            return video_path
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None

# Function to process video content
def process_video(video_path, user_query):
    try:
        # Simulate video processing (replace with actual video analysis logic)
        with st.spinner("Processing video and gathering insights..."):
            # Simulate a delay for processing
            time.sleep(5)
            
            # Generate a mock response
            response = f"Analysis of the video '{Path(video_path).name}' for the query: {user_query}\n\n"
            response += "The video appears to contain content related to AI and machine learning. Key insights include..."
            
            return response
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

# Handle source selection and processing
if source_type == "Web URL":
    # Web URL input
    url = st.sidebar.text_input("Enter website URL", "https://docs.smith.langchain.com/")
    if st.sidebar.button("Process Web Content"):
        st.session_state.vectors = process_web_content(url)
elif source_type == "PDF Upload":
    # PDF upload
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        # Check if the file is already processed
        file_name = uploaded_file.name
        if "last_processed_file" not in st.session_state or st.session_state.last_processed_file != file_name:
            st.session_state.vectors = process_pdf(uploaded_file)
            st.session_state.last_processed_file = file_name
elif source_type == "Video Summarizer":
    # Video summarizer section
    st.sidebar.header("Video Summarizer")
    youtube_url = st.sidebar.text_input(
        "Enter YouTube Video URL",
        placeholder="Paste the YouTube video link here",
        help="Provide a valid YouTube video URL."
    )
    video_file = st.sidebar.file_uploader(
        "Or upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis"
    )
    
    video_path = None
    
    if youtube_url:
        with st.spinner("Downloading YouTube video..."):
            video_path = download_youtube_video(youtube_url)
            if video_path:
                st.sidebar.success(f"Video downloaded successfully: {Path(video_path).name}")
                st.video(video_path, format="video/mp4", start_time=0)
    elif video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        st.video(video_path, format="video/mp4", start_time=0)

# Chat interface
st.header("Ask a question")
user_question = st.text_input("Enter your question:")

if user_question:
    if source_type == "Video Summarizer" and video_path:
        # Process video content
        response = process_video(video_path, user_question)
        if response:
            st.subheader("Video Analysis Result")
            st.write(response)
    elif "vectors" in st.session_state and st.session_state.vectors is not None:
        with st.spinner("Generating answer..."):
            # Create retrieval chain using the newer LangChain patterns
            retriever = st.session_state.vectors.as_retriever()
            
            # Build the RAG chain using the newer LCEL syntax
            rag_chain = (
                {"context": retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Track response time
            start = time.process_time()
            response = rag_chain.invoke(user_question)
            response_time = time.process_time() - start
            
            # Display answer
            st.subheader("Answer")
            st.write(response)
            st.caption(f"Response time: {response_time:.2f} seconds")
            
            # Show relevant document chunks
            with st.expander("Document Similarity Search"):
                # Get the documents that would be retrieved
                retrieved_docs = retriever.invoke(user_question)
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(doc.page_content)
                    st.write("--------------------------------")
    else:
        st.warning("Please process a web page, upload a PDF, or provide a video first!")

# Display app information
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This app demonstrates a Retrieval Augmented Generation (RAG) system using LangChain, Groq, and Ollama embeddings. It can process content from websites, PDF documents, or videos.")