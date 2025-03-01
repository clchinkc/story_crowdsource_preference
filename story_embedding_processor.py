import os
from dotenv import load_dotenv
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self, model_name='answerdotai/ModernBERT-base'):
        """Initialize the embedding processor with the specified model"""
        load_dotenv()
        
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not (supabase_url and supabase_key):
            raise ValueError("Missing Supabase credentials")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Load model and tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def _generate_embedding(self, text: str) -> list:
        """Generate embedding for a single text"""
        # Tokenize and move to device
        inputs = self.tokenizer(text, padding=True, truncation=True, 
                              max_length=2048, return_tensors='pt').to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        # Convert to list and return
        return embeddings[0].cpu().numpy().tolist()

    def process_pending_entries(self, batch_size=50, force_update=False):
        """Process entries that need embeddings
        
        Args:
            batch_size: Number of entries to process in each batch
            force_update: If True, update all embeddings regardless of existing values
        """
        try:
            # Get entries based on force_update parameter
            if force_update:
                response = self.supabase.table('feedback').select('*').execute()
                logger.info("Force update mode: Processing all entries")
            else:
                response = self.supabase.table('feedback').select('*').filter(
                    'embedding1', 'is', 'null'
                ).filter(
                    'embedding2', 'is', 'null'
                ).execute()
                logger.info("Normal mode: Processing only entries without embeddings")
            
            if not response.data:
                logger.info("No entries to process")
                return
            
            logger.info(f"Found {len(response.data)} entries to process")
            
            # Process in batches
            for i in range(0, len(response.data), batch_size):
                batch = response.data[i:i + batch_size]
                
                for entry in tqdm(batch, desc="Processing entries"):
                    updates = {}
                    
                    # Generate embeddings based on force_update
                    if force_update or not entry.get('embedding1'):
                        embedding1 = self._generate_embedding(entry['variation1'])
                        updates['embedding1'] = embedding1
                    
                    if force_update or not entry.get('embedding2'):
                        embedding2 = self._generate_embedding(entry['variation2'])
                        updates['embedding2'] = embedding2
                    
                    if updates:
                        # Update the entry with new embeddings
                        self.supabase.table('feedback').update(updates).eq(
                            'id', entry['id']
                        ).execute()
                        
            logger.info("Completed processing entries")
            
        except Exception as e:
            logger.error(f"Error processing entries: {str(e)}")
            raise

def main():
    try:
        processor = EmbeddingProcessor()
        # Add force_update parameter when calling the function
        processor.process_pending_entries(force_update=False)  # Change to True to force update all
    except Exception as e:
        logger.error(f"Failed to process embeddings: {str(e)}")

if __name__ == "__main__":
    main() 