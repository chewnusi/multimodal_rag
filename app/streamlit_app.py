import streamlit as st
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import search service
try:
    from search_service import MultimodalSearchService
except ImportError as e:
    st.error(f"Error importing search service: {e}")
    st.error("Make sure search_service.py is in the src/ folder")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_service' not in st.session_state:
    st.session_state.search_service = None

def initialize_search_service():
    """Initialize the search service with caching."""
    if st.session_state.search_service is None:
        with st.spinner("Initializing search service..."):
            try:
                st.session_state.search_service = MultimodalSearchService()
                # Load indexes
                text_loaded, image_loaded = st.session_state.search_service.load_all_indexes()
                
                if not text_loaded and not image_loaded:
                    st.error("No indexes found. Please create indexes first using create_indexes.py")
                    return False
                elif not text_loaded:
                    st.warning("Text index not found. Only image search will be available.")
                elif not image_loaded:
                    st.warning("Image index not found. Only text search will be available.")
                else:
                    st.success("Both text and image indexes loaded successfully!")
                
                return True
            except Exception as e:
                st.error(f"Error initializing search service: {e}")
                return False
    return True

def clear_query():
    """Clear the query and results."""
    st.session_state.query = ""
    st.session_state.search_results = None

def perform_search(max_results=3):
    """Perform search for the single query."""
    if not st.session_state.search_service:
        st.error("Search service not initialized")
        return
    
    query = st.session_state.query.strip()
    
    if not query:
        st.warning("Please enter a search query")
        return
    
    with st.spinner(f"Searching for '{query}'..."):
        try:
            # Search both text and images for the query
            results = st.session_state.search_service.search_multimodal(
                query=query,
                k_text=max_results,
                k_images=max_results
            )
            st.session_state.search_results = results
        except Exception as e:
            st.error(f"Error searching for '{query}': {e}")

def display_image_result(result: Dict[str, Any]):
    """Display a single image search result."""
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Extract the image path string from the result dict
            image_path = result['image_path']  # This should be a string
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
        possible_paths = [
            Path("data") / "img" / Path(image_path).name,    
            Path("../data") / "img" / Path(image_path).name,
        ]
        
        for path in possible_paths:
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
        # Title and metadata
        st.markdown(f"**{result['title']}**")
        st.markdown(f"*Article ID: {result['article_id']} | Date: {result['date'][:10] if result['date'] else 'N/A'}*")
        
        # Show associated images if any
        if result.get('images'):
            st.markdown("**Associated Images:**")
            
            # Display images in columns (max 3 per row)
            images = result['images']
            for i in range(0, len(images), 3):
                cols = st.columns(3)
                for j, img_path in enumerate(images[i:i+3]):
                    with cols[j]:
                        try:
                            # Try different path variations for images
                            working_path = find_working_image_path(img_path)
                            if working_path:
                                st.image(working_path, width='stretch')
                            else:
                                st.text(f"Image: {Path(img_path).name}")
                        except Exception as e:
                            st.text(f"Image: {Path(img_path).name}")
        
        # Article content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Toggle button for full/preview content
            show_full_key = f"show_full_{result['article_id']}"
            if show_full_key not in st.session_state:
                st.session_state[show_full_key] = False
            
            # Show content based on toggle state
            if st.session_state[show_full_key]:
                st.markdown(result['full_content'])
                button_text = "Show Preview"
            else:
                st.markdown(result['content'])
                button_text = "Show Full Article"
            
            # Toggle button
            if st.button(button_text, key=f"toggle_{result['article_id']}"):
                st.session_state[show_full_key] = not st.session_state[show_full_key]
                st.rerun()
        
        with col2:
            if result['url']:
                st.markdown(f"[📖 Read Original]({result['url']})")

def find_working_image_path(image_path: str) -> str:
    """Find working path for an image file."""
    possible_paths = [
        Path("data") / "img" / Path(image_path).name,
        Path("../data") / "img" / Path(image_path).name
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None

def display_search_results():
    """Display search results."""
    if not st.session_state.search_results:
        return
    
    st.header(f"Search Results for: '{st.session_state.query}'")
    
    # Create tabs for text and image results
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
    
    # Initialize search service
    if not initialize_search_service():
        st.stop()
    
    # Sidebar with system information
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
        
        # Search options
        st.header("Search Options")
        search_mode = st.selectbox(
            "Search Mode",
            ["Both Text & Images", "Text Only", "Images Only"]
        )
        
        max_results = st.slider("Max Results per Query", 1, 20, 3)
        
        if st.button("🔄 Reload Indexes"):
            st.session_state.search_service = None
            st.rerun()
    
    # Main search interface
    st.header("Search Query")
    st.markdown("Enter your search query to find relevant articles and images.")
    
    # Single query input
    st.session_state.query = st.text_input(
        "Search Query",
        value=st.session_state.query,
        placeholder="Enter your search query here...",
        key="main_query"
    )
    
    # Action buttons
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Start Search", type="primary", width='stretch'):
            perform_search(max_results)
    
    with col2:
        if st.button("🧹 Clear", width='stretch'):
            clear_query()
            st.rerun()
    
    # Display results
    if st.session_state.search_results:
        display_search_results()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Multimodal RAG System | Powered by Google Vertex AI & FAISS"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()