import streamlit as st
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

sys.path.append(str(Path(__file__).parent.parent / "src"))

from search_service import MultimodalSearchService
from answer_generation_service import AnswerGenerationService

st.set_page_config(
    page_title="Multimodal RAG Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'query' not in st.session_state:
    st.session_state.query = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_service' not in st.session_state:
    st.session_state.search_service = None
if 'generated_answer' not in st.session_state:
    st.session_state.generated_answer = None
if 'answer_service' not in st.session_state:
    st.session_state.answer_service = None

def initialize_search_service():
    """Initialize the search service with caching."""
    services_initialized = True
    
    if st.session_state.search_service is None:
        with st.spinner("Initializing search service..."):
            try:
                st.session_state.search_service = MultimodalSearchService()
                text_loaded, image_loaded = st.session_state.search_service.load_all_indexes()
                
                if not text_loaded and not image_loaded:
                    st.error("No indexes found. Please create indexes first using create_indexes.py")
                    return False
                elif not text_loaded:
                    st.warning("Text index not found. Only image search will be available.")
                elif not image_loaded:
                    st.warning("Image index not found. Only text search will be available.")
                    
            except Exception as e:
                st.error(f"Error initializing search service: {e}")
                services_initialized = False
    
    if st.session_state.answer_service is None:
        with st.spinner("Initializing answer generation service..."):
            try:
                st.session_state.answer_service = AnswerGenerationService()
            except Exception as e:
                st.error(f"Error initializing answer generation service: {e}")
                st.error("Make sure GOOGLE_API_KEY is set in your environment")
                services_initialized = False
    
    return services_initialized

def clear_query():
    """Clear the query and results."""
    st.session_state.query = ""
    st.session_state.search_results = None

def perform_search_and_generate_answer(max_results=4):
    """Perform search and generate answer for the query."""
    if not st.session_state.search_service or not st.session_state.answer_service:
        st.error("Services not initialized")
        return
    
    query = st.session_state.query.strip()
    
    if not query:
        st.warning("Please enter a search query")
        return
    
    with st.spinner(f"Searching for '{query}' and generating answer..."):
        try:
            results = st.session_state.search_service.search_multimodal(
                query=query,
                k_text=max_results,
                k_images=max_results
            )
            st.session_state.search_results = results
            
            llm_articles = min(max_results, 6)
            llm_images = min(max_results, 6)
            
            answer_result = st.session_state.answer_service.generate_answer_with_summary(
                query=query,
                search_results=results,
                n_articles=llm_articles,
                n_images=llm_images
            )
            st.session_state.generated_answer = answer_result
            
        except Exception as e:
            st.error(f"Error processing query '{query}': {e}")

def display_generated_answer():
    """Display the AI-generated answer."""
    if not st.session_state.generated_answer:
        return
    
    st.header("🤖 AI-Generated Answer")
    
    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color: #f0f8ff; 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 4px solid #4CAF50;
                margin-bottom: 20px;
            ">
                {st.session_state.generated_answer['answer']}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sources Used", 
                st.session_state.generated_answer['sources_used']
            )
        
        with col2:
            st.metric(
                "Primary Articles", 
                st.session_state.generated_answer['primary_articles']
            )
        
        with col3:
            st.metric(
                "Primary Images", 
                st.session_state.generated_answer['primary_images']
            )

def display_image_result(result: Dict[str, Any]):
    """Display a single image search result."""
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image_path = result['image_path']
            display_image_helper(image_path, caption=result['filename'])
                        
        with col2:
            st.markdown(f"**{result['title']}**")
            st.markdown(f"*Article ID: {result['article_id']}*")
            st.markdown(f"*Filename: {result['filename']}*")
            
            if result['url']:
                st.markdown(f"[📖 Read Original]({result['url']})")
            
            st.markdown(f"**Date:** {result['date'][:10] if result['date'] else 'N/A'}")

def display_image_helper(image_path: str, caption: str = ""):
    """Helper function to display images with path resolution."""
    try:
        path = Path("data") / "img" / Path(image_path).name
        
        if path.exists():
            st.image(str(path), caption=caption, width='stretch')
            return True
        
        st.error(f"Image not found: {Path(image_path).name}")
        return False
        
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.text(f"Attempted path: {image_path}")
        return False

def display_text_result(result: Dict[str, Any]):
    """Display a single text search result with associated images."""
    with st.container():
        st.markdown(f"**{result['title']}**")
        st.markdown(f"*Article ID: {result['article_id']} | Date: {result['date'][:10] if result['date'] else 'N/A'}*")
        
        if result.get('images'):
            st.markdown("**Associated Images:**")
            
            images = result['images']
            for i in range(0, len(images), 3):
                cols = st.columns(3)
                for j, img_path in enumerate(images[i:i+3]):
                    with cols[j]:
                        try:
                            working_path = find_working_image_path(img_path)
                            if working_path:
                                st.image(working_path, width='stretch')
                            else:
                                st.text(f"Image: {Path(img_path).name}")
                        except Exception as e:
                            st.text(f"Image: {Path(img_path).name}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            show_full_key = f"show_full_{result['article_id']}"
            if show_full_key not in st.session_state:
                st.session_state[show_full_key] = False
            
            if st.session_state[show_full_key]:
                st.markdown(result['full_content'])
                button_text = "Show Preview"
            else:
                st.markdown(result['content'])
                button_text = "Show Full Article"
            
            if st.button(button_text, key=f"toggle_{result['article_id']}"):
                st.session_state[show_full_key] = not st.session_state[show_full_key]
                st.rerun()
        
        with col2:
            if result['url']:
                st.markdown(f"[📖 Read Original]({result['url']})")

def find_working_image_path(image_path: str) -> str:
    """Find working path for an image file."""
    path = Path("data") / "img" / Path(image_path).name
    
    if path.exists():
        return str(path)
    return None

def display_search_results():
    """Display search results."""
    if not st.session_state.search_results:
        return
    
    st.header(f"Search Results for: '{st.session_state.query}'")
    
    text_tab, image_tab = st.tabs(["📝 Text Results", "🖼️ Image Results"])
    
    with text_tab:
        if st.session_state.search_results['text']:
            for i, result in enumerate(st.session_state.search_results['text']):
                display_text_result(result)
                if i < len(st.session_state.search_results['text']) - 1:
                    st.divider()
        else:
            st.info("No text results found")
    
    with image_tab:
        if st.session_state.search_results['images']:
            for i, result in enumerate(st.session_state.search_results['images']):
                display_image_result(result)
                if i < len(st.session_state.search_results['images']) - 1:
                    st.divider()
        else:
            st.info("No image results found")

def main():
    """Main Streamlit application."""
    st.title("🔍 Multimodal RAG Search")
    st.markdown("Search through articles using both text and images")
    
    if not initialize_search_service():
        st.stop()
    
    with st.sidebar:
        st.header("System Information")
        
        if st.session_state.search_service:
            stats = st.session_state.search_service.get_index_stats()
            
            if stats.get('text', {}).get('loaded'):
                st.success(f"✅ Text Index: {stats['text'].get('num_documents', 'N/A')} documents")
            else:
                st.error("❌ Text Index: Not loaded")
            
            if stats.get('images', {}).get('loaded'):
                st.success(f"✅ Image Index: {stats['images'].get('num_documents', 'N/A')} images")
            else:
                st.error("❌ Image Index: Not loaded")
        
        st.divider()
        
        max_results = st.slider("Max Results per Query", 1, 20, 3)
        st.caption("💡 **Tip:** You can retrieve up to 20 results for browsing, but the AI will only analyze the top 6 most relevant articles for generating answers")
        
        if st.button("🔄 Reload Indexes"):
            st.session_state.search_service = None
            st.rerun()
    
    st.header("Search Query")
    st.markdown("Enter your search query to get an AI-powered answer, based on found relevant articles and images.")
    
    st.session_state.query = st.text_input(
        "Search Query",
        value=st.session_state.query,
        placeholder="Enter your search query here...",
        key="main_query"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Search & Generate Answer", type="primary", width='stretch'):
            perform_search_and_generate_answer(max_results)
    
    with col2:
        if st.button("🧹 Clear", width='stretch'):
            clear_query()
            st.rerun()

    if st.session_state.generated_answer:
        display_generated_answer()
        st.divider()
    
    if st.session_state.search_results:
        display_search_results()
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Multimodal RAG System | Powered by Google Vertex AI & FAISS"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()