# Conversational AI with Memory and Topic Extraction

This repository contains a conversational AI system that maintains user profiles, extracts conversation topics, and generates personalized responses. The system is built using Python and leverages OpenAI's GPT models for natural language processing.

## Features

- **User Profile Management**: Maintains and updates user information including name, age, and interests
- **Topic Extraction**: Automatically identifies and clusters key topics from conversations
- **Personalized Responses**: Generates context-aware responses based on user profile and conversation history
- **Memory System**: Preserves conversation history and user preferences between sessions

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
.
├── Agents/
│   ├── personalizer.py      # User profile management
│   ├── response_generator.py # Response generation
│   ├── prompts.py          # System prompts
│   └── user_profile.json   # Stored user profiles
├── Topic_Extractor/
│   └── chat_topic_extractor.py # Topic extraction logic
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. Start chatting with the AI. The system will:
   - Extract and update your profile information
   - Maintain conversation context
   - Generate personalized responses
   - Extract topics when you exit (type 'exit' to end the session)

3. Your profile and conversation history will be saved automatically.

## Key Components

### MemoryAgent
- Manages user profiles
- Detects missing information
- Updates profile based on conversation

### ResponseGeneratorAgent
- Generates contextual responses
- Uses user profile for personalization
- Maintains conversation history

### Topic Extractor
- Identifies key topics from conversations
- Clusters related phrases
- Generates topic descriptions

## Customization

You can customize the system by:
1. Modifying prompts in `Agents/prompts.py`
2. Adjusting topic extraction parameters in `Topic_Extractor/chat_topic_extractor.py`
3. Updating the user profile model in `Agents/personalizer.py`

## Notes

- The system requires an active internet connection to use the OpenAI API
- User profiles and conversation history are stored locally in JSON format
- Topics are extracted and updated when you exit the conversation
