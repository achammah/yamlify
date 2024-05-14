import os
import json
import yaml
from dotenv import load_dotenv
from fetchURLContent import run_actor_and_fetch_data
from OpenAPIBuilder import process_parts_concurrently, generate_openapi_map, parse_openapi_schema, merge_yaml
from messageTemplate import test  # Assuming this is your placeholder for scrapped content

# Load environment variables from .env file
load_dotenv()

APIFY_API_KEY = os.getenv('APIFY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def fetch_and_save_url_content(api_key, start_url, output_file):
    content = run_actor_and_fetch_data(api_key, start_url)
    with open(output_file, 'w') as file:
        file.write(content)
    print(f"Content fetched from {start_url} and saved to {output_file}")

def load_documents(doc_file):
    with open(doc_file, 'r') as file:
        content = file.read()
    return chunk_string(content, 9000)

def load_documents_dict(doc_dict):
    content = json.dumps(doc_dict)
    return chunk_string(content, 9000)

def generate_openapi_from_url(api_key, url, output_file):
    fetch_and_save_url_content(api_key, url, 'scrapped_content.json')
    document_parts = load_documents('scrapped_content.json')
    openapi_schema = generate_openapi_map(url)
    parsed_schema = parse_openapi_schema(openapi_schema)
    process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
    print(f"OpenAPI schema generated and saved to {output_file}")

def generate_openapi_from_docs(doc_file, output_file):
    document_parts = load_documents(doc_file)
    with open('openapi_schema.json', 'r') as file:
        openapi_schema = json.load(file)
    parsed_schema = parse_openapi_schema(openapi_schema)
    process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
    print(f"OpenAPI schema generated and saved to {output_file}")

def generate_openapi_from_docs_dict(doc_dict, output_file):
    document_parts = load_documents_dict(doc_dict)
    with open('openapi_schema.json', 'r') as file:
        openapi_schema = json.load(file)
    parsed_schema = parse_openapi_schema(openapi_schema)
    process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
    print(f"OpenAPI schema generated and saved to {output_file}")

def generate_openapi_yaml_from_url(api_key, url, output_file):
    fetch_and_save_url_content(api_key, url, 'scrapped_content.json')
    generate_openapi_from_docs('scrapped_content.json', output_file)

def generate_openapi_yaml_from_docs(doc_file, output_file):
    generate_openapi_from_docs(doc_file, output_file)

def generate_openapi_yaml_from_docs_dict(doc_dict, output_file):
    generate_openapi_from_docs_dict(doc_dict, output_file)

def generate_openapi_yaml_from_map_and_docs(map_file, doc_file, output_file):
    with open(map_file, 'r') as file:
        openapi_schema = json.load(file)
    parsed_schema = parse_openapi_schema(openapi_schema)
    document_parts = load_documents(doc_file)
    process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
    print(f"OpenAPI schema generated and saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    # generate_openapi_yaml_from_url(APIFY_API_KEY, 'https://finnhub.io/docs/api/', 'final_openapi_schema.yaml')
    # generate_openapi_yaml_from_docs('scrapped_content.json', 'final_openapi_schema.yaml')
    # generate_openapi_yaml_from_docs_dict(test, 'final_openapi_schema.yaml')
    # generate_openapi_yaml_from_map_and_docs('openapi_schema.json', 'scrapped_content.json', 'final_openapi_schema.yaml')

    pass
