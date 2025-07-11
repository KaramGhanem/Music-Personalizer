from typing import List, Dict, Optional
from browserbase import Browserbase
from playwright.sync_api import sync_playwright
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID")

class BrowserAgent:
    def __init__(
        self,
        api_key: str = BROWSERBASE_API_KEY,
        project_id: str = BROWSERBASE_PROJECT_ID,
        proxy: bool = False
    ):
        """
        Initialize the BrowserAgent with Browserbase configuration.
        
        Args:
            api_key (str): Browserbase API key
            project_id (str): Browserbase project ID
            proxy (bool): Whether to use proxy for requests
        """
        if not api_key:
            raise ValueError("Browserbase API key is required")
        if not project_id:
            raise ValueError("Browserbase project ID is required")
            
        self.browserbase = Browserbase(api_key=api_key)
        self.project_id = project_id
        self.proxy = proxy
        self.session = None

    def create_session(self):
        """Create a new browser session."""
        session_params = {"project_id": self.project_id}
        if self.proxy:
            session_params["proxy"] = True
        self.session = self.browserbase.sessions.create(**session_params)
        return self.session

    def scrape_url(self, url: str, text_content: bool = True) -> Dict:
        """
        Scrape content from a URL using Browserbase.
        
        Args:
            url (str): The URL to scrape
            text_content (bool): Whether to return text content or HTML
            
        Returns:
            Dict: Dictionary containing the scraped content and metadata
        """
        if not self.session:
            self.create_session()

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(self.session.connect_url)
            context = browser.contexts[0]
            page = context.pages[0]

            # Navigate to URL
            page.goto(url)
            
            # Get content based on the text_content flag
            if text_content:
                content = page.inner_text("body")
            else:
                content = page.content()

            # Close browser
            page.close()
            browser.close()

            return {
                "content": content,
                "url": url,
                "metadata": {
                    "session_id": self.session.id,
                    "project_id": self.project_id
                }
            }

    def scrape_multiple_urls(self, urls: List[str], text_content: bool = True) -> List[Dict]:
        """
        Scrape multiple URLs using Browserbase.
        
        Args:
            urls (List[str]): List of URLs to scrape
            text_content (bool): Whether to return text content or HTML
            
        Returns:
            List[Dict]: List of dictionaries containing scraped content and metadata
        """
        results = []
        for url in urls:
            try:
                result = self.scrape_url(url, text_content)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "url": url,
                    "metadata": {
                        "session_id": self.session.id if self.session else None,
                        "project_id": self.project_id
                    }
                })
        return results

    def close_session(self):
        """Close the current browser session."""
        if self.session:
            self.session = None 