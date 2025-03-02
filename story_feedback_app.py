import streamlit as st
from story_dataset_generator import (
    StoryPrompt,
    get_configured_client,
    generate_variations,
)
import json
from datetime import datetime
from st_supabase_connection import SupabaseConnection

# Configure PROVIDERS_CONFIG to include all available models
PROVIDERS_CONFIG = {
    "prompt": {
        "provider": "github",
        "model": "openai/gpt-4o-mini"
    },
    "available_models": [
        {
            "provider": "gemini",
            "model": "gemini/gemini-2.0-flash-exp",
            "display_name": "Gemini 2.0 Flash"
        },
        {
            "provider": "azure",
            "model": "azure/gpt-4o-mini",
            "display_name": "Azure GPT-4o-mini"
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
                "base_prompt": feedback_data["prompt"].premise,
                "variation1": variations[i],
                "variation2": variations[j],
                "choice_source": feedback_data["choice_source"],
                "preferred_index": feedback_data["user_choice"],
                "evaluation_metadata": json.dumps({
                    "comparison_type": "direct_choice",
                    "model1": feedback_data["model_choices"][i],
                    "model2": feedback_data["model_choices"][j]
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

def generate_story_variations(prompt_text: str, model_choices: list):
    """Generate prompt and variations from user input using selected models"""
    try:
        # Initialize clients
        prompt_client = get_configured_client(PROVIDERS_CONFIG["prompt"]["provider"])
        
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
        
        variations = []
        # Generate a variation for each selected model
        for model_config in model_choices:
            variation_client = get_configured_client(model_config["provider"])
            variation = generate_variations(
                structured_prompt,
                1,  # Generate one variation per model
                variation_client,
                model_config["model"]
            )[0]  # Take first variation since we only generate one
            variations.append(variation)
        
        return structured_prompt, variations
    
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None, None

def convert_to_preference_format(feedback_entries: list) -> list:
    """Convert feedback entries to preference dataset format"""
    preference_dataset = []
    
    for entry in feedback_entries:
        # Create a preference pair
        preference_pair = {
            "chosen_conversations": [
                {
                    "content": entry["base_prompt"],
                    "role": "user"
                },
                {
                    "content": entry["variation1"] if entry["preferred_index"] == 1 else entry["variation2"],
                    "role": "assistant"
                }
            ],
            "rejected_conversations": [
                {
                    "content": entry["base_prompt"],
                    "role": "user"
                },
                {
                    "content": entry["variation2"] if entry["preferred_index"] == 1 else entry["variation1"],
                    "role": "assistant"
                }
            ]
        }
        preference_dataset.append(preference_pair)
    
    return preference_dataset

def download_preference_dataset():
    """Create and download preference dataset in DPO format"""
    try:
        # Get feedback entries from Supabase
        conn = st.connection("supabase", type=SupabaseConnection)
        result = conn.client.table('feedback').select("*").execute()
        
        if result.data:
            # Convert to preference format
            preference_dataset = convert_to_preference_format(result.data)
            
            # Convert to JSON string
            json_str = json.dumps(preference_dataset, indent=2)
            
            # Create download button
            st.download_button(
                label="Download DPO Dataset",
                data=json_str,
                file_name="story_preference_dataset.json",
                mime="application/json"
            )
            
            # Show preview
            st.info(f"Dataset contains {len(preference_dataset)} preference pairs")
            with st.expander("Preview Dataset"):
                st.json(preference_dataset[:3] if len(preference_dataset) > 3 else preference_dataset)
        else:
            st.warning("No feedback entries found to export")
            
    except Exception as e:
        st.error(f"Failed to generate preference dataset: {str(e)}")

def main():
    st.title("Story Variant Feedback Collector")
    st.markdown("### Help improve AI storytelling by choosing the better version!")
    
    # Add tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Generate & Compare", "Historical Feedback", "Export Dataset"])
    
    with tab1:
        # Initialize session state
        if "variations" not in st.session_state:
            st.session_state.variations = None
        if "prompt" not in st.session_state:
            st.session_state.prompt = None
        if "feedback_submitted" not in st.session_state:
            st.session_state.feedback_submitted = False
        if "model_choices" not in st.session_state:
            st.session_state.model_choices = None
        
        # Check if feedback was submitted in the previous interaction
        if st.session_state.feedback_submitted:
            st.success("Thank you for your feedback!")
            if st.button("Generate New Variations"):
                st.session_state.feedback_submitted = False
                st.session_state.variations = None
                st.rerun()
            return  # Stop further execution
        
        # User input
        user_prompt = st.text_area("Enter your story concept or theme:", 
                                    placeholder="e.g. 'A space janitor discovers a alien conspiracy'")
        
        # Model selection
        st.subheader("Select Models for Comparison")
        col1, col2 = st.columns(2)
        with col1:
            model1 = st.selectbox(
                "Model 1",
                options=PROVIDERS_CONFIG["available_models"],
                format_func=lambda x: x["display_name"],
                key="model1"
            )
        with col2:
            model2 = st.selectbox(
                "Model 2",
                options=PROVIDERS_CONFIG["available_models"],
                format_func=lambda x: x["display_name"],
                key="model2"
            )
        
        # Generation controls
        if st.button("Generate Story Variations"):
            with st.spinner("Creating story variants..."):
                model_choices = [model1, model2]
                prompt, variations = generate_story_variations(user_prompt, model_choices)
                if variations and len(variations) >= 2:
                    st.session_state.prompt = prompt
                    st.session_state.variations = variations
                    st.session_state.model_choices = model_choices
                    st.rerun()
        
        # Display variations
        if st.session_state.variations:
            st.markdown("---")
            st.markdown(f"**Prompt:** {st.session_state.prompt.premise}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Version 1 ({st.session_state.model_choices[0]['display_name']})")
                st.markdown(st.session_state.variations[0].content)
                if st.button("Choose Version 1 ➔", key="btn1"):
                    save_feedback({
                        "prompt": st.session_state.prompt,
                        "variations": [v.content for v in st.session_state.variations],
                        "user_choice": 1,
                        "choice_source": "human",
                        "model_choices": st.session_state.model_choices
                    })
                    st.session_state.feedback_submitted = True
                    st.rerun()
            
            with col2:
                st.subheader(f"Version 2 ({st.session_state.model_choices[1]['display_name']})")
                st.markdown(st.session_state.variations[1].content)
                if st.button("Choose Version 2 ➔", key="btn2"):
                    save_feedback({
                        "prompt": st.session_state.prompt,
                        "variations": [v.content for v in st.session_state.variations],
                        "user_choice": 2,
                        "choice_source": "human",
                        "model_choices": st.session_state.model_choices
                    })
                    st.session_state.feedback_submitted = True
                    st.rerun()

    with tab2:
        st.header("Historical Feedback")
        show_historical_feedback()
    
    with tab3:
        st.header("Export Preference Dataset")
        st.markdown("""
        Export the feedback database in a format compatible with Direct Preference Optimization (DPO).
        The dataset will contain pairs of chosen and rejected story variations for each prompt.
        """)
        download_preference_dataset()

def show_historical_feedback():
    conn = st.connection("supabase", type=SupabaseConnection)
    result = conn.client.table('feedback').select("*").order('created_at', desc=True).execute()
    
    if result.data:
        feedback_count = len(result.data)
        st.info(f"Total feedback entries in database: {feedback_count}")
        
        # Add a search/filter box
        search_term = st.text_input("Search prompts:", "")
        
        # Filter and display entries
        filtered_entries = [
            entry for entry in result.data 
            if search_term.lower() in entry['base_prompt'].lower()
        ] if search_term else result.data
        
        for entry in filtered_entries:
            with st.expander(f"Prompt: {entry['base_prompt'][:100]}...", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Version 1**")
                    st.markdown(entry['variation1'])
                with col2:
                    st.markdown("**Version 2**")
                    st.markdown(entry['variation2'])
                
                metadata = json.loads(entry['evaluation_metadata'])
                st.markdown(f"**Selected**: Version {entry['preferred_index']}")
                
                # Safely display model information if available
                if 'model1' in metadata and 'model2' in metadata:
                    st.markdown(f"**Models**: {metadata['model1']['display_name']} vs {metadata['model2']['display_name']}")
                elif 'comparison_type' in metadata:
                    st.markdown(f"**Comparison Type**: {metadata['comparison_type']}")
                
                st.markdown(f"**Date**: {entry['created_at'][:10]}")
    else:
        st.info("No feedback entries found in the database.")

if __name__ == "__main__":
    main() 