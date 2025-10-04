# Multimodal RAG System for AI News Articles

Retrieval-Augmented Generation (RAG) system that enables search and question-answering across both text content and images from AI news articles. The system scrapes articles from The Batch newsletter, creates searchable vector indexes, and provides AI-powered answers using Google's Gemini models.

---

## Project Overview

This system demonstrates a complete multimodal RAG pipeline that:
- **Scrapes** AI news articles and associated images from The Batch newsletter
- **Processes** both textual content and visual information into vector embeddings
- **Indexes** data using FAISS for efficient similarity search
- **Retrieves** relevant context from both text and images based on user queries
- **Generates** comprehensive answers using Google's Gemini 2.0 Flash model with vision capabilities

---

## System Architecture

### Technical Approach

The system implements a **bidirectional multimodal RAG architecture** where:

1. **Text Processing Pipeline**:
   - Articles are chunked and embedded using Google's `gemini-embedding-001` model
   - Text embeddings capture semantic meaning of article content
   - FAISS indexes enable fast similarity search

2. **Image Processing Pipeline**:
   - Images are embedded using Vertex AI's `multimodalembedding@001` model
   - Contextual text (article titles) enhance image embeddings
   - Separate FAISS index for visual content

3. **Retrieval Strategy**:
   - Parallel search across both text and image indexes
   - Results ranked by semantic similarity to query
   - Associated images linked to their source articles

4. **Answer Generation**:
   - Top-k relevant documents retrieved (default: 3 articles, 3 images)
   - Content sent to Gemini 2.0 Flash with vision capabilities
   - Model analyzes both textual and visual information
   - Generates comprehensive, well-structured answers

### Tools & Models Selected

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| **Web Scraping** | Scrapy | Robust, efficient crawler with built-in rate limiting and respect for robots.txt |
| **Text Embeddings** | Google Gemini Embedding (gemini-embedding-001) | High-quality 768-dimensional embeddings optimized for retrieval tasks |
| **Image Embeddings** | Vertex AI Multimodal Embedding (multimodalembedding@001) | 1408-dimensional embeddings supporting both visual and contextual text |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | Efficient similarity search, persistent storage, works well with LangChain |
| **LLM** | Google Gemini 2.0 Flash | Multimodal capabilities, fast inference, supports vision + text reasoning |
| **Framework** | LangChain | Simplifies integration between embeddings, vector stores, and LLMs |
| **UI** | Streamlit | Rapid development, interactive components, easy deployment |

### Design Decisions

**1. Separate Vector Stores for Text and Images**
- Improves embedding quality
- Allows different retrieval strategies per modality
- Simplifies maintenance and updates

**2. Contextual Text for Image Embeddings**
- Article titles provide semantic context for images
- Improves embedding quality and retrieval accuracy
- Bridges gap between visual and textual information

---

## Prerequisites

- **Python**: 3.10 or higher
- **Google Cloud Project**: With Vertex AI API enabled
- **Google API Key**: For Gemini models
- **Operating System**: Linux, macOS, or Windows (WSL recommended for Windows)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/chewnusi/multimodal_rag
cd multimodal_rag
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Google Cloud Project ID (for Vertex AI)
PROJECT_ID=your-google-cloud-project-id

# Required: Google API Key (for Gemini models)
GOOGLE_API_KEY=your-google-api-key

# Optional: LangSmith tracing (for debugging)
LANGSMITH_TRACING=false # set to true for enabling
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT=""
```

**Getting Your Credentials:**

1. **PROJECT_ID**: 
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create or select a project
   - Enable Vertex AI API: `gcloud services enable aiplatform.googleapis.com`
   - Set quota project: `gcloud auth application-default set-quota-project PROJECT_ID`

2. **GOOGLE_API_KEY**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key for Gemini models

### 4. Authenticate with Google Cloud

```bash
# Set up application default credentials
gcloud auth application-default login

# Set quota project (important for Vertex AI)
gcloud auth application-default set-quota-project PROJECT_ID
```

---

## Usage Guide

### Step 1: Scrape Articles (Optional)

The repository includes sample data, but you can scrape fresh articles:

```bash
cd batch_scraper
# Scrape specific number of articles
scrapy crawl batch_news -a max_articles=50 -s CLOSESPIDER_ITEMCOUNT=50
```

### Step 2: Create Vector Indexes (Optional)

Sample indexes are included, but you can recreate them from freshly scraped data:

```bash
# From project root
# Create both text and image indexes
python3 src/create_indexes.py

# Create only text index
python3 src/create_indexes.py --text-only

# Create only image index
python3 src/create_indexes.py --image-only

# Force recreate existing indexes
python3 src/create_indexes.py --force

# Check environment without creating indexes
python3 src/create_indexes.py --check

# Create in root folder or with custom paths
python3 src/create_indexes.py --data data --indexes indexes
```

**Important Note:**
- **Image Processing Time**: Expect 20-30 minutes for 200 images due to API rate limits

### Step 3: Run the Application

```bash
# From project root
streamlit run app/streamlit_app.py
```

---

## How It Works

### Search & Retrieval Flow

```
User Query
    ↓
[Query Embedding]
    ↓
    ├─→ [Text Vector Store] → Top-k Text Results
    └─→ [Image Vector Store] → Top-k Image Results
    ↓
[Result Merging & Linking]
    ↓
[Context Preparation]
    ↓
[Gemini 2.0 Flash + Vision]
    ↓
AI-Generated Answer
```