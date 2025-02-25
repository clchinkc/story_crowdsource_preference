import streamlit as st
from story_dataset_generator import (
    StoryPrompt,
    StoryContent,
    get_configured_client,
    generate_variations,
    StoryDataset
)
import json
from datetime import datetime
import os
from st_supabase_connection import SupabaseConnection

# Configure providers (same as generator config)
PROVIDERS_CONFIG = {
    "prompt": {
        "provider": "github",
        "model": "openai/gpt-4o-mini"
    },
    "variations": [
        {
            "provider": "gemini",
            "model": "gemini/gemini-2.0-flash-exp"
        }
    ]
}

def save_feedback(feedback_data: dict):
    """Save user feedback to Supabase database"""
    conn = st.connection("supabase", type=SupabaseConnection)
    
    # Create a record for each comparison pair
    variations = feedback_data["variations"]
    records = []
    
    # Generate all unique pairs
    for i in range(len(variations)):
        for j in range(i+1, len(variations)):
            records.append({
                "base_prompt": feedback_data["prompt"].premise,  # Store as string
                "variation1": variations[i],
                "variation2": variations[j],
                "choice_source": feedback_data["choice_source"],
                "preferred_index": feedback_data["user_choice"],
                "evaluation_metadata": json.dumps({
                    "comparison_type": "direct_choice"
                }),
                "created_at": datetime.now().isoformat()
            })
    
    # Use Supabase client directly
    if records:
        result = conn.client.table('feedback').insert(records).execute()
        if result.data:
            st.success(f"Saved {len(result.data)} feedback entries")
        else:
            st.error(f"Failed to save feedback: {result.error}")
    else:
        st.error("No valid comparisons to save")

def generate_story_variations(prompt_text: str):
    """Generate prompt and variations from user input"""
    try:
        # Initialize clients
        prompt_client = get_configured_client(PROVIDERS_CONFIG["prompt"]["provider"])
        variation_client = get_configured_client(PROVIDERS_CONFIG["variations"][0]["provider"])
        
        # Generate structured prompt
        structured_prompt = prompt_client.chat.completions.create(
            model=PROVIDERS_CONFIG["prompt"]["model"],
            messages=[{
                "role": "system",
                "content": f"""Convert this story idea into structured prompt:
                {prompt_text}
                Include title, genre, premise, main conflict, and protagonist. Make sure you stay true to the original story idea."""
            }],
            response_model=StoryPrompt,
            temperature=0.7
        )
        
        # Generate variations
        variations = generate_variations(
            structured_prompt,
            2,
            variation_client,
            PROVIDERS_CONFIG["variations"][0]["model"]
        )
        
        return structured_prompt, variations
    
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None, None

def main():
    st.title("Story Variant Feedback Collector")
    st.markdown("### Help improve AI storytelling by choosing the better version!")
    
    # Initialize session state
    if "variations" not in st.session_state:
        st.session_state.variations = None
    if "prompt" not in st.session_state:
        st.session_state.prompt = None
    
    # User input
    user_prompt = st.text_area("Enter your story concept or theme:", 
                                placeholder="e.g. 'A space janitor discovers a alien conspiracy'")
    
    # Generation controls
    if st.button("Generate Story Variations"):
        with st.spinner("Creating story variants..."):
            prompt, variations = generate_story_variations(user_prompt)
            if variations and len(variations) >= 2:
                st.session_state.prompt = prompt
                st.session_state.variations = variations
                st.rerun()
    
    # Display variations
    if st.session_state.variations:
        st.markdown("---")
        st.markdown(f"**Prompt:** {st.session_state.prompt.premise}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Version 1")
            st.markdown(st.session_state.variations[0].content)
            if st.button("Choose Version 1 ➔", key="btn1"):
                save_feedback({
                    "prompt": st.session_state.prompt,
                    "variations": [v.content for v in st.session_state.variations],
                    "user_choice": 1,
                    "choice_source": "human"
                })
                st.session_state.variations = None
                st.rerun()
        
        with col2:
            st.subheader("Version 2")
            st.markdown(st.session_state.variations[1].content)
            if st.button("Choose Version 2 ➔", key="btn2"):
                save_feedback({
                    "prompt": st.session_state.prompt,
                    "variations": [v.content for v in st.session_state.variations],
                    "user_choice": 2,
                    "choice_source": "human",
                })
                st.session_state.variations = None
                st.rerun()

    if st.button("View Historical Feedback"):
        show_historical_feedback()

def show_historical_feedback():
    conn = st.connection("supabase", type=SupabaseConnection)
    result = conn.client.table('feedback').select("*").execute()
    
    if result.data:
        st.subheader("Previous Feedback Entries")
        for entry in result.data:
            with st.expander(f"Feedback {entry['id']} - {entry['created_at']}"):
                st.json(entry)
    else:
        st.error(f"Error retrieving feedback: {result.error}")

if __name__ == "__main__":
    main() 