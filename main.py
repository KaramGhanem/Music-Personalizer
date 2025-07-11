from Agents.personalizer import MemoryAgent
import time
from typing import List
import logging
from Agents.response_generator import ResponseGeneratorAgent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
from Topic_Extractor.chat_topic_extractor import extract_topics
    

# Configure logging to track execution flow and timing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_queries(query: List[str], MA: MemoryAgent, RGA: ResponseGeneratorAgent, chat_history, tailored_profile):
    """
    Process a user query through the chatbot system.
    
    Args:
        query: List containing the user's input message
        MA: MemoryAgent instance for managing user profile
        RGA: ResponseGeneratorAgent instance for generating responses
        chat_history: List of previous conversation messages
        tailored_profile: Current user profile with personalization data
    
    Returns:
        tuple: (chatbot_response, updated_tailored_profile)
    """
    start_time = time.time()
    
    # Update the user profile with new information from the query
    # This will extract any new personal details and update the profile
    if tailored_profile:
        tailored_profile = MA.provide_tailored_profile(
            user_query=query,
            chat_history=chat_history,
            existing_user_profile=tailored_profile,
        )
    else:
        # If no profile exists, create a new one
        tailored_profile = MA.provide_tailored_profile(
            user_query=query,
            chat_history=chat_history,
            existing_user_profile=tailored_profile,
        )
            
    end_time = time.time()
    execution_time = end_time - start_time

    # Generate a response using the updated profile and conversation history
    chatbot_response = RGA.generate_response(
        query=query[0],  # Since query is a list, take the first element
        chat_history=chat_history,
        input_from_memory={"user_profile": tailored_profile}
    )
    
    # Log the interaction details for monitoring
    logging.info(f"Query: {query}")
    logging.info(f"Execution time: {execution_time:.2f} seconds")
    logging.info(f"Tailored Profile: {tailored_profile}")
    print(f"Response : {chatbot_response['response']}")
    logging.info("=" * 50)

    return chatbot_response, tailored_profile

def main():
    """
    Main function that initializes and runs the chatbot system.
    Handles the interactive chat loop and persistence of conversation data.
    """
    # Load environment variables from .env file
    load_dotenv(find_dotenv())
    
    # Retrieve the OpenAI API key from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if OPENAI_API_KEY is None:
        raise ValueError("Please set your OPENAI_API_KEY in a .env file or as an environment variable.")

    # Initialize the language model with specific parameters
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # Initialize the two main components of the system:
    # 1. MemoryAgent: Manages user profile and personalization
    # 2. ResponseGeneratorAgent: Generates contextual responses
    MA = MemoryAgent(llm=llm)
    RGA = ResponseGeneratorAgent()

    # Load any existing conversation history and user profile
    # This ensures continuity across chat sessions
    chat_history = RGA.load_conversation()
    tailored_profile = MA.load_profile()  # This will return None if no profile exists

    # Start the interactive chat loop
    print("\nWelcome to the chat! Type 'exit' to end the conversation.\n")
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit condition
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            # Extract topics from the entire conversation history before exiting
            topics_result = extract_topics(chat_history)
            if topics_result:
                topics, phrase_roles, rep_sentences = topics_result
                print("\nConversation Topics:")
                for topic_id, topic_info in topics.items():
                    print(f"Topic {topic_id}:")
                    print(f"  Title: {topic_info['Topic Title']}")
                    print(f"  Description: {topic_info['Topic Description']}")
                # Save the topics in the profile
                MA.save_profile(
                    profile=tailored_profile['user_profile'],
                    current_note=tailored_profile['note'],
                    topics=topics
                )
            break
            
        # Process the user's input and get a response
        chatbot_response, tailored_profile = process_queries([user_input], MA, RGA, chat_history, tailored_profile)
        
        # Update the conversation history with the new interaction
        chat_history = chatbot_response.get('conversation_history', [])
        
        # Save the updated conversation and profile after each message
        # This ensures data persistence between sessions
        RGA.save_conversation()
        MA.save_profile(
            profile=tailored_profile['user_profile'],
            current_note=tailored_profile['note'],
            topics=tailored_profile.get('topics')  # Preserve existing topics if any
        )

if __name__ == "__main__":
    main()

    

