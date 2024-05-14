import os
import json
from dotenv import load_dotenv
from apify_client import ApifyClient

# Load environment variables from .env file
load_dotenv()

def run_actor_and_fetch_data(api_key, start_url):
    # Initialize the Apify client with your API token
    client = ApifyClient(api_key)

    # Define the actor input according to your needs
    run_input = {
        "startUrls": [{"url": start_url}],
        "proxyConfiguration": {"useApifyProxy": True},
        "crawlerType": "playwright:firefox",
        "maxCrawlDepth": 0,
        "initialConcurrency": 10,
        "maxConcurrency": 100,
        "saveHtml": False,
        "saveMarkdown": True,
        "removeElementsCssSelector": "nav, footer, script, style, noscript, svg, [role='alert'], [role='banner'], [role='dialog'], [role='alertdialog'], [role='region'][aria-label*='skip' i], [aria-modal='true']",
        "clickElementsCssSelector": "[aria-expanded='false']",
        "removeCookieWarnings": True
    }

    # Start the actor and wait for it to finish
    actor_call = client.actor('apify~website-content-crawler').call(run_input=run_input)

    # Fetch results from the actor's default dataset
    dataset = client.dataset(actor_call['defaultDatasetId']).list_items()
    return json.dumps(dataset.items)  # Convert dataset items to JSON and return