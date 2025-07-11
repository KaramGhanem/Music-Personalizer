import logging
import warnings
import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


logging.getLogger().setLevel(logging.INFO)
warnings.simplefilter("ignore", category=DeprecationWarning)

#
# ------------------ 1. PYDANTIC BaseModels ------------------
#

class UserProfile(BaseModel):
    """
    Extended user-profile data model for a musical AI assistant that generates chord progressions.
    Captures musical context, stylistic preferences, skill level, and user goals to personalize responses.
    """

    name: Optional[str] = Field(
        None,
        description="Extract the user's own name if explicitly mentioned. If the user mentions someone else's name, do not extract."
    )
    
    age: Optional[int] = Field(
        None,
        description="Extract the user's own age if explicitly mentioned. If the user mentions another person's age, ignore it."
    )
    
    interests: Optional[List[str]] = Field(
        None,
        description="Extract the user's own goals, plans or interests only if the user is referring to themselves. Present as concise phrases."
    )

    musical_goals: Optional[List[str]] = Field(
        None,
        description="User's personal goals in music, e.g., 'write emotional ballads', 'improve jazz harmony', 'produce lofi beats'."
    )
    
    favorite_genres: Optional[List[str]] = Field(
        None,
        description="Genres the user prefers working with or listening to, such as jazz, pop, orchestral, lofi, rock, R&B."
    )

    playing_instruments: Optional[List[str]] = Field(
        None,
        description="Instruments the user plays or uses to compose (e.g., piano, guitar, DAW, synths). Used to adapt technical complexity."
    )

    musical_skill_level: Optional[str] = Field(
        None,
        description="User's self-declared skill level with music composition or harmony theory. Examples: 'beginner', 'intermediate', 'advanced'."
    )

    composition_context: Optional[str] = Field(
        None,
        description="Context or use-case for composition, such as 'film scoring', 'songwriting for myself', 'working with a band', or 'TikTok music'."
    )

    emotion_or_mood_preferences: Optional[List[str]] = Field(
        None,
        description="User’s preferred emotional/mood tone in music: e.g., dreamy, melancholic, epic, uplifting, tense."
    )

    rejection_of_guided_discovery: Optional[bool] = Field(
        None,
        description="Set to True if the user explicitly rejects attempts to extract more info or follow-up questions."
    )

    keywords: Optional[List[str]] = Field(
        None,
        description="Keywords that describe the genre, style, song type, key, mode, and bar to generate a chord progression."
    )


class MemoryAgent(object):
    """
    Manages user profile information in memory, detects missing info,
    and merges new user details from the conversation.
    """
    def __init__(self, llm):
        """
        Args:
            llm: An LLM client with .invoke(prompt_str: str) -> response.
            user_profile: An existing UserProfile instance, if any.
        """
        self.llm = llm
        self._logger = logging.getLogger(__name__)
        self.profile_file = "user_profile.json"  # Initialize the profile file path
        self._setup_profile_parser_chain()

    def _setup_profile_parser_chain(self):
        """
        Prepares the chain for extracting name, age, interests, and goals 
        from a user query using a PydanticOutputParser.
        """
        extract_information_prompt = """
        You only need to extract *the user's* name, age, interests and goals — i.e., information about the speaker themself. 
        If the user is describing someone else, ignore that information and do not extract it. 
        
        Output valid JSON only.
        
        User Input: {query}
        
        {format_instructions}
        """

        parser = PydanticOutputParser(pydantic_object=UserProfile)
        prompt = ChatPromptTemplate.from_template(
            extract_information_prompt,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        # Create a chain that: prompt -> llm -> pydantic parser
        self._profile_parser_chain = prompt | self.llm | parser

    
    def update_general_user_profile(
        self, 
        user_query: str,
        user_profile: Optional[UserProfile] = None,
    ) -> Dict[str, Any]:
        """
        Uses the chain to parse user_query and updates the in-memory user profile.

        Returns:
            Dict[str, Any]: The updated user profile as a dictionary.
        """
        # If user_profile is a dict, convert it:
        if isinstance(user_profile, dict):
            user_profile = UserProfile(**user_profile)
        elif user_profile is None:
            user_profile = UserProfile()
        
        # Directly invoke the chain
        parsed_profile: UserProfile = self._profile_parser_chain.invoke({"query": user_query})

        # Merge fields
        for field_name, new_value in parsed_profile.dict(exclude_unset=True).items():
            if new_value is not None:
                if field_name == "interests" and isinstance(new_value, list):
                    # If user_profile.goals is None, just assign the new list
                    if user_profile.interests is None:
                        user_profile.interests = new_value
                    else:
                        # Combine the old + new, removing duplicates if desired
                        combined_interests = user_profile.interests + new_value
                        user_profile.interests = list(combined_interests) # Append unique values
                else:
                    # For other fields (strings, ints, or lists not named "interests"), just overwrite
                    setattr(user_profile, field_name, new_value)
    
        # Return a dict version
        return user_profile.dict()

    
    def detect_missing_info(self, user_profile: Optional[dict]) -> Optional[str]:
        """
        Detects missing information in the user profile and returns an appropriate inquiry note.
    
        Args:
            user_profile (dict, optional): Dictionary containing user profile details.
    
        Returns:
            str or None: An inquiry note if fields are missing, otherwise None.
        """
        if not user_profile or all(value is None for value in user_profile.values()):
            return "User Profile is empty. No user information has been extracted yet. It might be helpful to ask the user about their name first."
    
        # Identify missing fields
        missing_fields = [field for field, value in user_profile.items() if value is None]
    
        if missing_fields:
            return (f"Missing field(s) in the user profile: {' , '.join(missing_fields)}. "
                    f"Consider asking the user about the missing details naturally in conversation. Try to predict what type of Keywords the user wants to generate a progression based on, using the user's preferences and previous progressions i.e. suggest to the user Keywords that describe the genre, style, song type, key, mode, and bar combination based on the user's preferences and conversation history to generate a chord progression.")  
    
        return ("Learn more about the user and their goals, plans or interests, and personalize the response accordingly. Continue with the user's chain of thought!"
                "Try to predict what type of Keywords the user wants to generate a progression based on, using the user's preferences and previous progressions i.e. suggest to the user Keywords that describe the genre, style, song type, key, mode, and bar combination based on the user's preferences and conversation history to generate a chord progression.")  

    def save_profile(self, profile, current_note=None, topics=None):
        """
        Save the user profile to a JSON file.
        
        Args:
            profile: The user profile data to save
            current_note: Optional note about missing information
            topics: Optional dictionary of extracted topics
        """
        # Convert topics dictionary keys to strings if topics exist
        if topics:
            topics = {str(k): v for k, v in topics.items()}
            
        profile_data = {
            "user_profile": profile,
            "note": current_note,
            "topics": topics
        }
        
        with open(self.profile_file, 'w') as f:
            json.dump(profile_data, f, indent=4)
            
    def load_profile(self):
        """
        Load the user profile from a JSON file.
        
        Returns:
            dict: The loaded profile data or None if file doesn't exist
        """
        try:
            with open(self.profile_file, 'r') as f:
                data = json.load(f)
                # Ensure the data has the correct structure
                if isinstance(data, dict):
                    if 'user_profile' in data:
                        return data
                    return {"user_profile": data, "note": None, "topics": None}
                return {"user_profile": data, "note": None, "topics": None}
        except FileNotFoundError:
            return None

    def provide_tailored_profile(
        self, 
        user_query: str,
        chat_history: Optional[List[Dict]] = None,
        existing_user_profile: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        1) Updates user profile with new info from user_query.
        2) Optionally checks for missing info in the conversation context.
        3) Returns the updated profile plus any note about missing info.

        Args:
            user_query (str): The user's latest message/input.
            chat_history (List[Dict[str, Any]]): The conversation so far.

        Returns:
            dict: The final user profile with optional "note" about missing info.
        """
        memory_message = {}
        # Update the profile with this new user input
        memory_message["user_profile"] = self.update_general_user_profile(
            user_query=user_query, 
            user_profile=existing_user_profile.get('user_profile') if existing_user_profile else None
        )
        
        # Preserve topics from existing profile
        if existing_user_profile and 'topics' in existing_user_profile:
            memory_message["topics"] = existing_user_profile["topics"]
        
        missing_info_note = self.detect_missing_info(
            user_profile=memory_message["user_profile"],
        )
        if missing_info_note:
            memory_message["note"] = missing_info_note

        # Save the updated profile with the current note
        self.save_profile(
            profile=memory_message["user_profile"],
            current_note=memory_message.get("note"),
            topics=memory_message.get("topics")
        )
        
        return memory_message