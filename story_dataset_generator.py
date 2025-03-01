import instructor
from litellm import completion
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import json
import logging
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
from supabase import create_client, Client
import torch
from transformers import AutoTokenizer
from story_ranking_dataset import StoryRewardModel

# Load environment variables for LLM API keys
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoryPrompt(BaseModel):
    """Data model for a story prompt"""
    title: str
    genre: str
    premise: str
    main_conflict: str
    protagonist: str

class StoryContent(BaseModel):
    """Data model for a single story variation content"""
    content: str = Field(..., description="The text content of the story variation")

class StoryComparison(BaseModel):
    variation1: str
    variation2: str
    reasoning: str
    preferred_index: int = Field(..., ge=1, le=2)
    source: str = Field(default="model", description="Origin of comparison: 'model' or 'human'")

class LLMComparison(BaseModel):
    """Data model for the LLM response during variation comparison.
    Only includes the reasoning and the preferred index.
    """
    reasoning: str
    preferred_index: int = Field(..., ge=1, le=2)

class StoryEvaluation(BaseModel):
    compared_pairs: List[StoryComparison]
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class StoryDataset(BaseModel):
    base_prompt: StoryPrompt
    variations: List[StoryContent] = Field(..., min_items=2)
    model_metadata: dict
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    evaluation: Optional[StoryEvaluation] = None
    
    @field_validator('variations')
    @classmethod
    def validate_variations(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 variations required")
        return v

# Update client initialization with validation
def get_configured_client(provider: str) -> instructor.Instructor:
    """Initialize and validate LLM client configuration"""
    clients = {
        "github": (os.getenv("GITHUB_TOKEN"), "https://models.inference.ai.azure.com"),
        "gemini": (os.getenv("GEMINI_API_KEY"), None),
        "azure": (os.getenv("AZURE_UST_SECONDARY_KEY"), "https://hkust.azure-api.net")
    }
    
    api_key, base_url = clients.get(provider, (None, None))
    if not api_key:
        raise ValueError(f"Missing API key for {provider}")

    return instructor.from_litellm(
        completion,
        api_key=api_key,
        api_base=base_url,
        api_version="2024-10-01-preview" if provider == "azure" else None
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def generate_story_prompts(client: instructor.Instructor, model: str, num_prompts: int) -> List[StoryPrompt]:
    """
    Generate multiple story prompts in a single LLM call
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a creative writing assistant. Generate {num_prompts} distinct story prompts. 
                    Each prompt must include title, genre, premise, main conflict, and protagonist. 
                    Ensure prompts are varied and include unexpected elements."""
                },
                {
                    "role": "user",
                    "content": f"Create {num_prompts} random story prompts for a creative writing exercise. Make them distinct and include surprising elements."
                }
            ],
            response_model=List[StoryPrompt],
            max_tokens=2000 * num_prompts,
            temperature=1.0,
            strict=True,
        )
        return response
    except Exception as e:
        logger.error(f"Error generating prompts: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def generate_variations(prompt: StoryPrompt, num_variations: int, client: instructor.Instructor, model: str) -> List[StoryContent]:
    """
    Generate variations of a story prompt using specified LLM with retry logic
    """
    try:
        system_message = f"""You are a creative editor. Based on this story concept:

        Title: {prompt.title}
        Genre: {prompt.genre}
        Premise: {prompt.premise}
        Main Conflict: {prompt.main_conflict}
        Protagonist: {prompt.protagonist}

        Generate {num_variations} unique variations of the story content. Each variation should:
        - Have a different take on the story while maintaining the core concept
        - Be returned as an object with a 'content' field
        - Contain new and interesting content only (no metadata, don't include the title, genre, premise, main conflict, or protagonist)
        - Be in JSON format with only the content text

        Return a JSON array of {num_variations} variation objects."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": system_message
                }
            ],
            response_model=List[StoryContent],
            max_tokens=5000 * num_variations,
            temperature=0.5,
            strict=True,
        )
        return response
    except Exception as e:
        logger.error(f"Error generating variations: {e}")
        raise

def load_reward_model(model_path='story_reward_model.pth'):
    """Load trained reward model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = StoryRewardModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    return model, tokenizer, device

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def compare_variations(variations: List[str], client: instructor.Instructor, model: str, 
                      reward_model, tokenizer, device, confidence_threshold: float = 0.2) -> StoryEvaluation:
    """
    Compare variations using custom model first, only query LLM for uncertain pairs
    """
    try:
        comparisons = []
        # Compare all unique pairs
        for i in range(len(variations)):
            for j in range(i+1, len(variations)):
                var1 = variations[i]
                var2 = variations[j]
                
                # Get custom model prediction
                inputs = tokenizer(
                    [var1, var2],
                    max_length=2048,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                with torch.no_grad():
                    scores = reward_model(inputs['input_ids'], inputs['attention_mask'])
                    scores = scores.view(-1, 2)  # Reshape to [batch_size, 2]
                    logits = scores[:, 0] - scores[:, 1]  # Compare scores directly
                    score_diff = abs(logits[0].item())  # Use logit difference as confidence
                    preferred_index = 1 if logits[0] > 0 else 2  # Positive logit means prefer first option
                
                logger.info(f"Pair {i+1} vs {j+1} - Confidence score: {score_diff:.3f}")
                
                if score_diff >= confidence_threshold:
                    # Use custom model judgment
                    logger.info(f"Using reward model (confidence above {confidence_threshold})")
                    comparison = StoryComparison(
                        variation1=var1,
                        variation2=var2,
                        reasoning=f"Custom model preference with confidence {score_diff:.3f}",
                        preferred_index=preferred_index,
                        source="model"  # Using reward model
                    )
                else:
                    # Fall back to LLM judgment
                    logger.info(f"Using LLM (low confidence: {score_diff:.3f})")
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "system",
                            "content": f"""Act as expert story editor. Compare these variations:
                            Option 1: {var1}
                            Option 2: {var2}
                            
                            Consider: creativity, coherence, character potential, market appeal.
                            Provide detailed reasoning and choose the better one (1 or 2)."""
                        }],
                        response_model=LLMComparison,
                        max_tokens=2000,
                        temperature=0.2,
                    )
                    
                    comparison = StoryComparison(
                        variation1=var1,
                        variation2=var2,
                        reasoning=f"LLM judgment: {response.reasoning}",
                        preferred_index=response.preferred_index,
                        source="llm"  # Using LLM
                    )
                
                comparisons.append(comparison)
        
        return StoryEvaluation(
            compared_pairs=comparisons,
            evaluated_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        raise

def save_dataset_to_supabase(dataset: StoryDataset):
    """Save generated dataset to existing feedback table"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        
        records = []
        
        # Create records for all comparison pairs
        for comparison in dataset.evaluation.compared_pairs:
            records.append({
                "base_prompt": dataset.base_prompt.premise,
                "variation1": comparison.variation1,
                "variation2": comparison.variation2,
                "choice_source": comparison.source,
                "preferred_index": comparison.preferred_index,
                "evaluation_metadata": json.dumps({
                    "model_metadata": dataset.model_metadata,
                    "reasoning": comparison.reasoning
                }),
                "created_at": dataset.generated_at
            })
        
        if records:
            result = supabase.table('feedback').insert(records).execute()
            if result.data:
                logger.info(f"Saved {len(result.data)} comparisons")
                return True
                
    except Exception as e:
        logger.error(f"Supabase save failed: {str(e)}")
        return False

def generate_batch_dataset(
    num_prompts: int = 1,
    variations_per_prompt: int = 2,
    providers_config: dict = None
) -> None:
    """Generate a batch of story datasets"""
    if providers_config is None:
        raise ValueError("providers_config must be provided")
    
    try:
        # Load reward model once per batch
        reward_model, tokenizer, device = load_reward_model()
        
        # Generate all prompts in a single call
        prompt_client = get_configured_client(providers_config["prompt"]["provider"])
        prompts = generate_story_prompts(
            prompt_client, 
            providers_config["prompt"]["model"],
            num_prompts
        )
        
        if not prompts:
            logger.error("No prompts generated")
            return
            
        for prompt in prompts:
            try:
                # Generate variations using multiple models
                all_variations = []
                variation_models = []
                num_models = len(providers_config["variations"])
                variations_per_model = max(1, variations_per_prompt // num_models)
                remaining_variations = variations_per_prompt - (variations_per_model * num_models)
                
                for i, variation_config in enumerate(providers_config["variations"]):
                    current_variations = variations_per_model + (1 if i < remaining_variations else 0)
                    
                    variation_client = get_configured_client(variation_config["provider"])
                    variations = generate_variations(
                        prompt,
                        current_variations,
                        variation_client,
                        variation_config["model"]
                    )
                    all_variations.extend(variations)
                    variation_models.append(variation_config["model"])
                
                if all_variations:
                    # Add evaluation step
                    evaluation = None
                    if providers_config.get("evaluation"):
                        eval_client = get_configured_client(providers_config["evaluation"]["provider"])
                        evaluation = compare_variations(
                            [v.content for v in all_variations],
                            eval_client,
                            providers_config["evaluation"]["model"],
                            reward_model,
                            tokenizer,
                            device
                        )
                    
                    dataset = StoryDataset(
                        base_prompt=prompt,
                        variations=all_variations,
                        model_metadata={
                            "base_model": providers_config["prompt"]["model"],
                            "variation_models": variation_models,
                            "evaluation_model": providers_config.get("evaluation", {}).get("model")
                        },
                        evaluation=evaluation
                    )
                    
                    # Save only to Supabase
                    save_dataset_to_supabase(dataset)
                    
            except Exception as e:
                logger.error(f"Failed to generate dataset iteration: {e}")
                continue
    except Exception as e:
        logger.error(f"Failed to generate batch dataset: {e}")

if __name__ == "__main__":
    providers_config = {
        "prompt": {
            "provider": "github",
            "model": "openai/gpt-4o-mini"
        },
        "variations": [
            {
                "provider": "gemini",
                "model": "gemini/gemini-2.0-flash-exp"
            },
            {
                "provider": "azure",
                "model": "azure/gpt-4o-mini"
            }
        ],
        "evaluation": {
            "provider": "azure",
            "model": "azure/gpt-4o-mini"
        }
    }
    
    # Generation configuration
    config = {
        "num_prompts": 2,
        "variations_per_prompt": 2,
        "providers_config": providers_config
    }
    
    generate_batch_dataset(**config) 