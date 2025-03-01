# Story Crowdsource Preference System

🌟 **We're creating an open source dataset for story preferences! Star this repository to be notified when we release it.** 🌟

Try it now: [Story Preference Collection App](https://storycrowdsourcepreference.streamlit.app)

A comprehensive system for collecting, analyzing, and learning from story preferences using AI models and human feedback. This project combines multiple AI models with human feedback to improve story generation and preference learning.

## Overview

The system consists of several key components:

1. **Story Generation**: Uses multiple AI models (GPT-4, Gemini, etc.) to generate story variations from prompts
2. **Feedback Collection**: Web interface for collecting human preferences between story variations
3. **Embedding Processing**: Generates and stores embeddings for story variations
4. **Reward Model Training**: Trains a reward model based on collected preferences
5. **Dataset Generation**: Creates preference datasets suitable for Direct Preference Optimization (DPO)

## Key Highlights

- **Open Source Dataset**: We will open source the dataset here in this repository, making it a valuable resource for both the tech and writer communities.
- **Live Demo**: Try our [Story Preference Collection App](https://storycrowdsourcepreference.streamlit.app) to contribute your preferences
- **Streamlit + Supabase**: The system is built using Streamlit for the web interface and Supabase for data management.
- **Benchmarking Tool**: This dataset will be instrumental in benchmarking Large Language Models (LLMs) for creative writing.

## Features

- Multi-model story generation with configurable provider selection
- Interactive web interface for story comparison and feedback collection
- Automated embedding generation for story variations
- Customizable reward model training with source weighting
- Export capabilities for DPO-compatible datasets
- Supabase integration for data storage and management

## Participate and Contribute

- **⭐ Star the Repository**: Your first step! Star this repository to show your support and be notified when we release the dataset.
- **Input Your Preferences**: Visit our [Story Preference Collection App](https://storycrowdsourcepreference.streamlit.app) to input your story preferences and contribute to the dataset.

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (optional, but recommended for training)
- Supabase account and project
- API keys for supported AI models (GPT-4, Gemini, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/story_crowdsource_preference.git
cd story_crowdsource_preference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following variables:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GITHUB_TOKEN=your_github_token
GEMINI_API_KEY=your_gemini_api_key
AZURE_UST_SECONDARY_KEY=your_azure_key
```

## Components

### Story Dataset Generator (`story_dataset_generator.py`)
- Generates story prompts and variations using multiple AI models
- Handles story comparison and evaluation
- Integrates with Supabase for data storage

### Feedback Collection App (`story_feedback_app.py`)
- Streamlit-based web interface
- Allows users to compare and rate story variations
- Exports collected preferences in DPO format

### Embedding Processor (`story_embedding_processor.py`)
- Processes story variations to generate embeddings
- Uses ModernBERT for embedding generation
- Manages embedding storage in Supabase

### Reward Model Training (`story_ranking_dataset.py`)
- Implements preference learning from collected feedback
- Supports weighted training based on feedback source
- Includes model evaluation and testing capabilities

## Usage

1. Start the feedback collection app:
```bash
streamlit run story_feedback_app.py
```

2. Generate story datasets:
```bash
python story_dataset_generator.py
```

3. Process embeddings:
```bash
python story_embedding_processor.py
```

4. Train the reward model:
```bash
python story_ranking_dataset.py
```

## Configuration

### Provider Configuration
The system supports multiple AI providers with configurable weights and models:

```python
PROVIDERS_CONFIG = {
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
    ]
}
```

### Training Configuration
Customize reward model training parameters:

```python
config = {
    'batch_size': 4,
    'num_epochs': 5,
    'learning_rate': 3e-5,
    'test_size': 0.2,
    'source_weights': {
        'model': 1.0,   # Full weight to reward model feedback
        'llm': 0.5,     # Half weight to LLM feedback
        'human': 2.0    # Double weight to human feedback
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- ModernBERT by AnswerDotAI
- Streamlit for the web interface
- Supabase for database services
 
