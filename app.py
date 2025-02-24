import streamlit as st
from firecrawl import FirecrawlApp
import faiss
import openai
import os
import pickle
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv  # Import to load environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# Create the 'db' folder if it doesn't exist
if not os.path.exists("db"):
    os.makedirs("db")

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

# Define paths for embeddings and FAISS index
EMBEDDINGS_PATH = "db/index.pkl"
FAISS_INDEX_PATH = "db/index.faiss"

# Function to load embeddings
def load_embeddings(file_path):
    if not os.path.exists(file_path):
        print("Embeddings file not found. Starting with an empty list.")
        return []
    with open(file_path, "rb") as f:
        raw_embeddings = pickle.load(f)
        print(f"Raw embeddings loaded: {raw_embeddings}")
        
        # Validate and clean the embeddings list
        embeddings = []
        expected_dimension = None
        
        for i, item in enumerate(raw_embeddings):
            url = item.get("url", None)
            text = item.get("text", None)
            embedding = item.get("embedding", None)
            
            # Check if all required fields are present and valid
            if not url or not text or not isinstance(embedding, list) or len(embedding) == 0:
                print(f"Invalid item at index {i}: {item}")
                continue
            
            # Dynamically determine the expected embedding dimension
            if expected_dimension is None:
                expected_dimension = len(embedding)
                print(f"Determined embedding dimension: {expected_dimension}")
            
            # Ensure all embeddings have the same dimension
            if len(embedding) != expected_dimension:
                print(f"Invalid embedding dimension at index {i}: {len(embedding)} (expected {expected_dimension})")
                continue
            
            # Add valid items to the cleaned embeddings list
            embeddings.append({
                "url": url,
                "text": text,
                "embedding": embedding
            })
        
        print(f"Cleaned embeddings list: {embeddings}")
        return embeddings

# Function to load FAISS index
def load_faiss_index(file_path):
    if not os.path.exists(file_path):
        print("FAISS index file not found. Creating a new index.")
        return faiss.IndexFlatL2(1536)
    index = faiss.read_index(file_path)
    print(f"Loaded FAISS index with size: {index.ntotal}")
    
    # Debugging: Ensure the FAISS index matches the embeddings list
    global embeddings
    if index.ntotal != len(embeddings):
        print(f"Warning: FAISS index size ({index.ntotal}) does not match embeddings list size ({len(embeddings)}).")
    
    return index

# Build the RAG chain
def build_chain():
    global embeddings
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Debugging: Print the embeddings list
    print("Embeddings list:")
    for i, item in enumerate(embeddings):
        print(f"Item {i}: {item}")
    
    # Ensure embeddings list is properly structured
    if not embeddings or not all("text" in item and "embedding" in item and "url" in item for item in embeddings):
        st.error("Embeddings file is improperly structured or empty. Cannot build QA chain.")
        return None
    
    # Debugging: Print the number of valid embeddings
    print(f"Number of valid embeddings: {len(embeddings)}")
    
    # Dynamically determine the embedding dimension
    expected_dimension = len(embeddings[0]["embedding"])
    print(f"Determined embedding dimension: {expected_dimension}")
    
    # Create FAISS vector store from embeddings
    try:
        db = FAISS.from_embeddings(
            text_embeddings=[(item["text"], item["embedding"]) for item in embeddings],
            embedding=embeddings_model,
            metadatas=[{"url": item["url"]} for item in embeddings]
        )
        print("FAISS vector store created successfully.")
    except Exception as e:
        st.error(f"Failed to create FAISS vector store: {str(e)}")
        return None
    
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.5,
        max_tokens=500  # Reduce max_tokens to avoid exceeding the context limit
    )
    
    template = """You are a helpful assistant that answers questions based on the content of web pages.
    Context: {context}
    Question: {question}
    """
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),  # Fetch only the top 2 most relevant chunks
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    
    return qa

# Function to scrape URLs and generate embeddings
def scrape_urls(urls):
    global embeddings, index, qa_chain
    firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50 
    )
    
    for url in urls:
        try:
            st.info(f"Scraping URL: {url}")
            scraped_data = firecrawl_app.scrape_url(url=url, params={'formats': ['markdown']})
            content = scraped_data.get('markdown', '')
            
            if not content:
                st.warning(f"No content found for URL: {url}")
                continue
            
            # Split the content into smaller chunks
            chunks = text_splitter.split_text(content)
            
            # Generate embeddings for each chunk
            st.info(f"Generating embeddings for URL: {url}")
            for chunk in chunks:
                embedding = OpenAIEmbeddings().embed_query(chunk)
                
                # Add embedding to FAISS index
                embedding_np = np.array(embedding).astype('float32')
                index.add(np.array([embedding_np]))
                
                # Store the chunk and embedding
                new_entry = {
                    "url": url,
                    "text": chunk,
                    "embedding": embedding
                }
                embeddings.append(new_entry)
            
            # Save the updated embeddings and FAISS index
            with open(EMBEDDINGS_PATH, "wb") as f:
                pickle.dump(embeddings, f)
            faiss.write_index(index, FAISS_INDEX_PATH)
            st.success(f"Successfully processed URL: {url}")
        
        except Exception as e:
            st.error(f"Failed to process URL {url}: {str(e)}")
    
    # Rebuild the QA chain after updating embeddings
    st.info("Rebuilding QA chain...")
    global qa_chain
    qa_chain = build_chain()
    if qa_chain is None:
        st.error("Failed to rebuild the QA chain.")
    else:
        st.success("QA chain rebuilt successfully!")

# Function to ask a question
def ask_question(question):
    global qa_chain
    
    if qa_chain is None:
        st.error("No data available. Please scrape some URLs first.")
        return None, []
    
    try:
        st.info("Running QA chain...")
        result = qa_chain.invoke({"query": question})
        
        # Debugging: Print the retrieved context
        print("Retrieved context:")
        for doc in result['source_documents']:
            print(doc.page_content)
        
        response_text = result['result']
        sources = result['source_documents']
        
        if not response_text:
            st.warning("No answer generated by the QA chain.")
            return None, []
        
        # Format the sources
        formatted_sources = [{"url": doc.metadata.get("url", "Unknown")} for doc in sources]
        
        return response_text, formatted_sources
    
    except Exception as e:
        st.error(f"Failed to process the question: {str(e)}")
        return None, []

# Initialize global variables
embeddings = load_embeddings(EMBEDDINGS_PATH)
index = load_faiss_index(FAISS_INDEX_PATH)
qa_chain = build_chain()

# Streamlit App
st.title("Website Content Scraper and Question Answering System")

# Sidebar for scraping URLs
st.sidebar.header("Scrape Websites")
urls_input = st.sidebar.text_area("Enter URLs (one per line):")
if st.sidebar.button("Scrape URLs"):
    urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
    if urls:
        scrape_urls(urls)
        st.sidebar.success("URLs processed successfully!")
    else:
        st.sidebar.warning("Please enter at least one URL.")

# Main section for asking questions
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        answer, sources = ask_question(question)
        if answer:
            st.subheader("Answer:")
            st.write(answer)
            st.subheader("Sources:")
            for source in sources:
                st.write(f"- [{source['url']}]({source['url']})")
        else:
            st.info("No answer available. Please check the question or scrape more URLs.")