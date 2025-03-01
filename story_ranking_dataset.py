import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import random

class StoryPairDataset(Dataset):
    def __init__(self, tokenizer_name='answerdotai/ModernBERT-base', max_length=2048,
                 source_weights={'model': 1.0, 'llm': 1.0, 'human': 1.0}):
        """
        Initialize dataset with configurable source weights
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Max sequence length for tokenization
            source_weights: Dict of weights for each feedback source (0.0 to disable, 1.0 for full inclusion)
                          e.g. {'model': 1.0, 'llm': 0.5, 'human': 2.0}
        """
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Check for required environment variables
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "Missing required environment variables. Please ensure both "
                "SUPABASE_URL and SUPABASE_KEY are set in your environment or .env file."
            )
        
        # Initialize Supabase client
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Supabase client: {str(e)}")
        
        # Load data from Supabase feedback table
        try:
            feedback_response = supabase.table("feedback").select("*").execute()
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data from Supabase: {str(e)}")
        
        if not feedback_response.data:
            print("Warning: No data retrieved from feedback table")
            return
            
        # Group entries by source
        source_groups = {'model': [], 'llm': [], 'human': []}
        
        for entry in feedback_response.data:
            try:
                source = entry['choice_source']
                if source in source_weights and source_weights[source] > 0:
                    source_groups[source].append({
                        'variation1': entry['variation1'],
                        'variation2': entry['variation2'],
                        'preferred_index': entry['preferred_index'],
                        'source': source
                    })
            except Exception as e:
                print(f"Error processing entry {entry.get('id', 'unknown')}: {str(e)}")
        
        # Calculate number of samples to take from each source
        max_samples = max(len(group) for group in source_groups.values() if len(group) > 0)
        
        for source, weight in source_weights.items():
            if weight <= 0 or not source_groups[source]:
                continue
                
            # Calculate number of samples based on weight
            n_samples = int(max_samples * weight)
            
            # Repeat or sample entries to match desired proportion
            source_entries = source_groups[source]
            if n_samples <= len(source_entries):
                # Randomly sample if we need fewer entries
                sampled = random.sample(source_entries, n_samples)
            else:
                # Repeat entries if we need more
                repeats = n_samples // len(source_entries)
                remainder = n_samples % len(source_entries)
                sampled = source_entries * repeats + random.sample(source_entries, remainder)
            
            self.data.extend(sampled)
        
        # Shuffle the combined dataset
        random.shuffle(self.data)
        
        print(f"Dataset composition:")
        for source in source_weights:
            count = sum(1 for item in self.data if item['source'] == source)
            print(f"{source}: {count} samples ({count/len(self.data)*100:.1f}%)")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pair = self.data[idx]
        return self._tokenize_pair(pair['variation1'], pair['variation2'], pair['preferred_index'])
    
    def _tokenize_pair(self, text_a, text_b, label):
        encoding = self.tokenizer(
            [text_a, text_b],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long)
        }

class StoryRewardModel(nn.Module):
    def __init__(self, bert_model_name='answerdotai/ModernBERT-base'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # Freeze ModernBERT parameters to prevent fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = False
        self.regressor = nn.Sequential(
            nn.Conv1d(
                in_channels=self.bert.config.hidden_size,
                out_channels=128,
                kernel_size=4,
                padding=1
            ),
            nn.AdaptiveMaxPool1d(1),  # Reduces sequence length to 1
            nn.Flatten(),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states.permute(0, 2, 1)
        return self.regressor(hidden_states).squeeze(-1)

    def save(self, path='story_reward_model.pth'):
        torch.save(self.regressor.state_dict(), path)
        
    @classmethod
    def load(cls, path='story_reward_model.pth'):
        model = cls()  # This initializes a fresh BERT model
        # Check if file exists before loading
        if os.path.exists(path):
            # Only load the regressor parameters
            model.regressor.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            print(f"Warning: No saved model found at {path}. Using a fresh model.")
        return model

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Convert labels from 1/2 to 0/1
        labels = labels - 1  # Now 0 or 1
        
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        scores = model(flat_input_ids, flat_attention_mask)
        scores = scores.view(-1, 2)
        
        # Calculate logits as difference between scores
        logits = scores[:, 0] - scores[:, 1]
        
        loss = loss_fn(logits, labels.float())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Convert labels from 1/2 to 0/1
            labels = labels - 1  # Now 0 or 1
            
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
            flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            
            scores = model(flat_input_ids, flat_attention_mask)
            scores = scores.view(-1, 2)
            
            logits = scores[:, 0] - scores[:, 1]
            loss = loss_fn(logits, labels.float())
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = (logits > 0).long()  # Positive logit means preference for first option
            correct += preds.eq(labels).sum().item()
            
    return total_loss / len(dataloader), correct

def train_and_evaluate():
    config = {
        'batch_size': 4,
        'num_epochs': 5,
        'learning_rate': 3e-5,
        'test_size': 0.2,
        'warmup_ratio': 0.1,
        'margin': 0.5,
        'source_weights': {
            'model': 1.0,  # Full weight to reward model feedback
            'llm': 0.5,    # Half weight to LLM feedback
            'human': 2.0   # Double weight to human feedback
        },
        'model_path': 'story_reward_model.pth',
        'continue_training': True  # Set to True to load and continue training existing model
    }
    
    device = get_device()
    print(f"Using device: {device}")
    
    try:
        dataset = StoryPairDataset(source_weights=config['source_weights'])
    except (ValueError, ConnectionError) as e:
        print(f"Error loading dataset: {e}")
        print("Please check your environment variables and Supabase connection.")
        return
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Please ensure there is data in the feedback table.")
        return

    train_size = int((1 - config['test_size']) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size']*2)

    # Initialize or load model
    if config['continue_training'] and os.path.exists(config['model_path']):
        print(f"Loading existing model from {config['model_path']}")
        model = StoryRewardModel.load(config['model_path']).to(device)
    else:
        print("Initializing new model")
        model = StoryRewardModel().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * config['warmup_ratio']),
        num_training_steps=total_steps
    )

    loss_fn = nn.BCEWithLogitsLoss()
    
    best_accuracy = 0.0
    best_regressor_state = None
    
    for epoch in range(config['num_epochs']):
        avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        avg_test_loss, total_correct = evaluate_epoch(model, test_loader, loss_fn, device)
        accuracy = total_correct / len(test_dataset)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {accuracy:.2%} | Test Samples: {len(test_dataset)}")
        
        # Save best model's regressor state only
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_regressor_state = model.regressor.state_dict().copy()
            print(f"New best accuracy: {best_accuracy:.2%}")

    # Save the best regressor state only
    if best_regressor_state is not None:
        torch.save(best_regressor_state, config['model_path'])
    else:
        # If we never found a "best" model, save the final one
        torch.save(model.regressor.state_dict(), config['model_path'])
    print(f"\nTraining complete. Best model saved with accuracy: {best_accuracy:.2%}")
    
    model = model.to(device)

def test_model(model_path='story_reward_model.pth'):
    device = get_device()
    model = StoryRewardModel()
    
    # Check if the model file exists
    if os.path.exists(model_path):
        # Load only the regressor parameters
        regressor_state_dict = torch.load(model_path, map_location='cpu')
        model.regressor.load_state_dict(regressor_state_dict)
        print(f"Loaded model from {model_path}")
    else:
        raise ValueError(f"Model file not found at {model_path}")
    
    model = model.to(device)
    model.eval()
    
    test_pairs = [
        ("A young hero discovers magical powers and saves the world", 
         "A retired farmer solves a local mystery using wisdom gained from years of experience"),
        ("In a dystopian future, rebels fight against an oppressive regime",
         "A small-town baker discovers an ancient family recipe with mysterious properties")
    ]
    
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    
    print("\nTesting trained model:")
    with torch.no_grad():
        for i, (text_a, text_b) in enumerate(test_pairs, 1):
            encoding = tokenizer(
                [text_a, text_b],
                padding='max_length',
                truncation=True,
                max_length=2048,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            scores = model(input_ids.view(-1, input_ids.size(-1)),
                        attention_mask.view(-1, attention_mask.size(-1)))
            scores = scores.view(-1, 2)
            
            print(f"\nTest Pair {i}:")
            print(f"Story A: {text_a}")
            print(f"Story B: {text_b}")
            print(f"Scores - A: {scores[0, 0]:.2f}, B: {scores[0, 1]:.2f}")
            print(f"Model prefers: {'A' if scores[0, 0] > scores[0, 1] else 'B'}")

if __name__ == "__main__":
    train_and_evaluate()
    print("\nTesting model...")
    test_model() 