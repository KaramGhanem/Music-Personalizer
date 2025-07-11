#############################
# ResponseGenerator
#############################
import os
import logging
import json
from .prompts import direct_response_prompt, direct_and_inquiry_response_prompt
from typing import List, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.CRITICAL)

# Load environment variables from .env file
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("Please set your OPENAI_API_KEY in a .env file or as an environment variable.")
gpt_4o = ChatOpenAI(
    verbose=True,
    temperature=0.4, 
    model='gpt-4o',
    top_p=1,
    frequency_penalty=0,
    presence_penalty=1,
    stream=False,
    api_key=os.getenv("OPENAI_API_KEY")
)


class ResponseGeneratorAgent:
    def __init__(
            self, 
            llm: ChatOpenAI = gpt_4o
        ):
        """
        Initializes the ResponseGenerator with necessary inputs.
        """
        # Define the full message-based prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_query}\n\nUser Profile: {user_profile}\n\nRecent Conversation: {recent_conversation}\n\nInquiry Note: {inquiry_note}\n\nTopics: {topics}")
        ])

        self.chain = self.prompt_template | llm | StrOutputParser()

        # Initialize conversation history
        self.conversation_history = []

    def generate_response(
        self,
        query: str,
        chat_history: List[dict] = [{}],
        input_from_memory: dict = {},
    ):
        recent_conversation = chat_history[-4:] if len(chat_history) > 3 else chat_history
        user_profile = input_from_memory.get("user_profile", "N/A")['user_profile']
        inquiry_note = input_from_memory.get("user_profile", "")['note']
        topics = "N/A"
        
        # Safely check if user_profile exists and has topics
        profile_data = input_from_memory.get("user_profile", {})
        if isinstance(profile_data, dict) and profile_data.get('topics') is not None:
            topics = profile_data['topics']

        system_prompt = direct_response_prompt if not inquiry_note else direct_and_inquiry_response_prompt

        # Prepare input dict
        inputs = {
            "system_prompt": system_prompt,
            "user_query": query,
            "user_profile": str(user_profile) if isinstance(user_profile, dict) else user_profile,
            "recent_conversation": str(recent_conversation),
            "inquiry_note": inquiry_note,
            "topics": str(topics) if isinstance(topics, dict) else topics
        }

        response = self.chain.invoke(inputs)

        # Add the interaction to conversation history
        self.conversation_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])

        return {
            'response': response,
            'inquiry_note': inquiry_note,
            'conversation_history': self.conversation_history
        }

    def save_conversation(self, filename: str = "conversation.json"):
        """Save the conversation history to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=4, ensure_ascii=False)

    def load_conversation(self, filename: str = "conversation.json"):
        """Load conversation history from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            return self.conversation_history
        except FileNotFoundError:
            logging.info(f"No existing conversation file found at {filename}")
            return []
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {filename}")
            return []
