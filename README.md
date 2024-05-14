# YAMLify: Turn URL into OpenAPI Schema

YAMLify is a project that scrapes website content and converts it into an OpenAPI schema. This tool leverages Apify for web scraping and OpenAI for generating OpenAPI specifications from the scraped content.

## Overview

This project consists of the following main components:

1. **Web Scraping with Apify**: Fetches website content using the Apify platform.
2. **Schema Generation**: Generates OpenAPI schema parts from the scraped content.
3. **Schema Merging and Fixing**: Merges generated schema parts and fixes common schema errors.
4. **OpenAPI Schema Parsing**: Parses and validates the final OpenAPI schema.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/YAMLify.git
    cd YAMLify
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a `.env` file in the project root directory.
    - Add your Apify and OpenAI API keys to the `.env` file:
        ```ini
        APIFY_API_KEY=your_apify_api_key
        OPENAI_API_KEY=your_openai_api_key
        ```

## Components

### 1. Web Scraping with Apify

The `fetchURLContent.py` script scrapes website content using the Apify platform.


### 2. Schema Generation

The `messageTemplate.py` script includes all the prompts needed to interact with the OpenAI API for generating OpenAPI schema parts. Ensure you have the necessary environment variables set for your OpenAI key.

### 3. Schema Merging and Fixing

The `merge_yaml` function in the `OpenAPIBuilder.py` script merges the generated schema parts and fixes common schema errors.

### 4. OpenAPI Schema Parsing

The `parse_openapi_schema` function in the `OpenAPIBuilder.py` script parses and validates the final OpenAPI schema.

## How It Works

### Main Functions in OpenAPIBuilder.py

1. **process_parts_concurrently**: This function processes the document parts concurrently. It uses a thread pool executor to handle parts of the schema in parallel, ensuring efficient processing.

2. **generate_openapi_map**: This function orchestrates the entire process. It splits the document content into manageable parts, processes each part, and integrates the results into the OpenAPI schema.

3. **generate_openapi_part**: This function generates a specific part of the OpenAPI schema by interacting with the OpenAI API. It uses predefined prompts to structure the API calls and processes the responses to create schema parts.

4. **merge_yaml**: This function merges the different parts of the OpenAPI schema into a single YAML file. It handles the recursive merging of dictionaries and lists, ensuring no duplicate entries.

5. **yaml_cleanup**: This function cleans up the YAML structure by removing unwanted keys and empty entries.

6. **fix_schema_errors**: This function identifies and fixes common errors in the OpenAPI schema, such as invalid references and missing properties.

### Steps to Generate OpenAPI Schema

1. **Run the Web Scraping Script**:
    ```bash
    python3 fetchURLContent.py <start_url>
    ```
    Replace `<start_url>` with the URL of the website you want to scrape.

2. **Generate OpenAPI Schema Parts**:
    Use the OpenAI API and prompts defined in `messageTemplate.py` to generate schema parts.

3. **Merge and Fix Schema**:
    ```bash
    python3 OpenAPIBuilder.py
    ```

    This will merge the schema parts, fix common errors, and validate the final schema.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.
