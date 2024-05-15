import os
import json
from typing import List, Dict

from dotenv import load_dotenv
from fetchURLContent import run_actor_and_fetch_data
from OpenAPIBuilder import (
    process_parts_concurrently, generate_openapi_map, parse_openapi_schema, chunk_string
)
from messageTemplate import test  # Assuming this is your placeholder for scrapped content
from logger import Logger

yamlify_logger = Logger(name="yamlify", level=Logger.INFO)

# Load environment variables from .env file
load_dotenv()

APIFY_API_KEY = os.getenv('APIFY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# ensure env secrets assigned
assert APIFY_API_KEY
assert OPENAI_API_KEY


def read_json_from_file(filename: str) -> Dict:
    with open(filename, 'r') as file:
        openapi_schema = json.load(file)

    return openapi_schema


def fetch_and_save_url_content(api_key: str, start_url: str, output_filename: str):
    """Fetches content from `start_url` and saves it to `output_filename`."""
    content = run_actor_and_fetch_data(api_key, start_url)
    with open(output_filename, 'w') as file:
        file.write(content)
    yamlify_logger.info(f"Content fetched from {start_url} and saved to {output_filename}")


def load_documents(doc_file: str) -> List[str]:
    """Loads documents from the file `doc_file` and splits them into a list of chunks."""
    with open(doc_file, 'r') as file:
        content = file.read()
    chunks = chunk_string(content, 9000)
    return chunks


def load_documents_dict(doc_dict: Dict) -> List[str]:
    """Loads documents from the dictionary and splits them into a list of chunks."""
    content = json.dumps(doc_dict)
    chunks = chunk_string(content, 9000)
    return chunks


def generate_openapi_from_url(api_key: str, url: str, output_filename: str):
    """ Fetches content from url and generates openapi schema """
    temp_file_name = 'scrapped_content.json'
    fetch_and_save_url_content(api_key, url, temp_file_name)
    document_parts = load_documents(temp_file_name)
    openapi_schema = generate_openapi_map(url)
    parsed_schema = parse_openapi_schema(openapi_schema)
    process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
    yamlify_logger.info(f"OpenAPI schema generated and saved to {output_filename}")


def generate_openapi_from_docs(doc_file: str, output_filename: str):
    """ Generates openapi schema from provided filename `doc_file` """
    document_parts = load_documents(doc_file)
    openapi_schema = read_json_from_file('openapi_schema.json')
    parsed_schema = parse_openapi_schema(openapi_schema)
    process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
    yamlify_logger.info(f"OpenAPI schema generated and saved to {output_filename}")


def generate_openapi_from_docs_dict(doc_dict: Dict, output_filename: str):
    """ Generates openapi schema from provided doc_dict """
    document_parts = load_documents_dict(doc_dict)
    openapi_schema = read_json_from_file('openapi_schema.json')
    parsed_schema = parse_openapi_schema(openapi_schema)
    process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
    yamlify_logger.info(f"OpenAPI schema generated and saved to {output_filename}")


def generate_openapi_yaml_from_url(api_key: str, url: str, output_filename: str):
    """ Fetches content from `url` and generates openapi schema from it """
    fetch_and_save_url_content(api_key, url, 'scrapped_content.json')
    generate_openapi_from_docs('scrapped_content.json', output_filename)


def generate_openapi_yaml_from_map_and_docs(map_filename: str, doc_filename: str, output_filename: str):
    openapi_schema = read_json_from_file(map_filename)
    parsed_schema = parse_openapi_schema(openapi_schema)
    document_parts = load_documents(doc_filename)
    process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
    yamlify_logger.info(f"OpenAPI schema generated and saved to {output_filename}")


if __name__ == "__main__":
    # Example usage
    # generate_openapi_yaml_from_url(APIFY_API_KEY, 'https://finnhub.io/docs/api/', 'final_openapi_schema.yaml')
    # generate_openapi_from_docs('scrapped_content.json', 'final_openapi_schema.yaml')
    generate_openapi_from_docs_dict(test, 'final_openapi_schema.yaml')
    # generate_openapi_yaml_from_map_and_docs('openapi_schema.json', 'scrapped_content.json', 'final_openapi_schema.yaml')
