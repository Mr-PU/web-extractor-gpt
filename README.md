# Website Content Scraper and Question Answering System

Welcome to the Website Content Scraper and Question Answering System! This application is designed to scrape content from websites, generate embeddings using OpenAI's text-embedding model, and provide a question-answering system powered by a Retrieval-Augmented Generation (RAG) chain. The system leverages Streamlit for the user interface, Firecrawl for web scraping, and FAISS for efficient vector storage and retrieval.

## Features
- **Web Scraping**: Scrape content from multiple URLs and extract text in markdown format.
- **Embedding Generation**: Generate embeddings for the scraped content using OpenAI's text-embedding-ada-002 model.
- **Vector Storage**: Store embeddings in a FAISS index for fast and efficient similarity search.
- **Question Answering**: Ask questions about the scraped content and get answers powered by OpenAI's GPT-4o-mini model.
- **Source Attribution**: View the sources (URLs) of the content used to generate the answers.
- **Persistent Storage**: Save embeddings and FAISS index to disk for future use.

## Prerequisites
Before running the application, ensure you have the following:

- Python 3.8 or higher installed on your system.
- **API Keys:**
  - OpenAI API Key: Sign up at OpenAI and get your API key.
  - Firecrawl API Key: Sign up at Firecrawl and get your API key.
- Git installed (optional, for cloning the repository).

## Installation and Setup

### Step 1: Clone or Download the Repository

You can either clone the repository using Git or download it as a ZIP file.

#### Option 1: Clone the Repository
```bash
git clone https://github.com/Mr-PU/web-extractor-gpt.git
cd web-extractor-gpt
```

#### Option 2: Download as ZIP
- Click the **Code** button on the GitHub repository page and select **Download ZIP**.
- Extract the ZIP file to your desired location.
- Open a terminal and navigate to the extracted folder:
```bash
cd path/to/extracted-folder
```

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies. Here's how to create and activate one:

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

- Rename `.env.example` to `.env`:
```bash
mv .env.example .env
```

- Create a `.env` file in the root directory of the project and add your API keys:
```plaintext
OPENAI_API_KEY=your_openai_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

### Step 5: Run the Application

Start the Streamlit application by running the following command:
```bash
streamlit run app.py
```

Once the application is running, open your browser and navigate to the URL provided in the terminal (usually http://localhost:8501).

## How to Use the Application

### Scrape Websites:
- Enter the URLs of the websites you want to scrape in the sidebar.
- Click the **Scrape URLs** button to start scraping and generating embeddings.

### Ask Questions:
- Enter your question in the main input box.
- Click the **Get Answer** button to retrieve the answer and view the sources.

## Folder Structure
```
website-scraper-qa-system/
├── db/                    # Folder for storing embeddings and FAISS index
│   ├── index.pkl          # Serialized embeddings
│   └── index.faiss        # FAISS index file
├── app.py                 # Main application script
├── requirements.txt       # List of dependencies
├── README.md              # This file
└── .env                   # Environment variables (API keys)
```

## Technologies Used
- **Streamlit**: For building the user interface.
- **Firecrawl**: For web scraping and content extraction.
- **OpenAI**: For generating embeddings and powering the question-answering system.
- **FAISS**: For efficient vector storage and retrieval.
- **LangChain**: For building the RAG chain and managing the QA pipeline.

## Troubleshooting

### API Key Errors:
- Ensure that the `.env` file is correctly formatted and contains valid API keys.
- Double-check that the keys are active and have sufficient quota.

### FAISS Index Errors:
- If the FAISS index becomes corrupted, delete the `db/index.faiss` and `db/index.pkl` files and restart the application to create a new index.

### Scraping Failures:
- Ensure that the URLs are valid and accessible.
- Check the Firecrawl API key and quota.
