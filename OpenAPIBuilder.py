import os
import json
from typing import List, Dict, Union, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from messageTemplate import (
    humanMessage1,
    aiMessage1,
    test,
    systemMessageMapper,
    systemMessagePart
)
import concurrent.futures
from openapi_schema_validator import validate, OAS30Validator
from jsonschema import ValidationError
from deepdiff import DeepDiff
from logger import Logger

yamlify_logger = Logger(name="yamlify", level=Logger.INFO)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
assert OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


def process_parts_concurrently(document_parts: List[str], openapi_schema: Dict,
                               parsed_schema: Union[List, Dict]):
    parts_dict = {
        "info": "",
        "paths": {},
        "components": {},
        "security": "",
        "tags": ""
    }
    part_name_mapping = {
        'info': 'header',
        'tags': 'footer'
    }

    def process_individual_part(part_name: str, part_data: Union[str, Dict]) -> Union[Tuple[None, None], Tuple[str, Dict]]:
        try:
            mapped_part_name = part_name_mapping.get(part_name, part_name)
            yamlify_logger.info(f'Processing part: {mapped_part_name}')
            # part_data here is expected to be a dictionary with details necessary for processing the part
            return part_name, generate_openapi_part(document_parts, openapi_schema, mapped_part_name, part_data)
        except Exception as exc:
            yamlify_logger.exception(f'Error processing part: {exc}')
            return None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        # Handle header
        futures.append(executor.submit(process_individual_part, 'info',
                                       {"title": parsed_schema["title"], "servers": parsed_schema["servers"]}))
        # Handle paths
        for path_key in parsed_schema["paths"]:
            futures.append(
                executor.submit(process_individual_part, 'paths', {path_key: parsed_schema["paths"][path_key]}))
        # Handle components
        for component_key in parsed_schema["components"]:
            futures.append(executor.submit(process_individual_part, 'components',
                                           {component_key: parsed_schema["components"][component_key]}))
        # Handle security
        for security_item in parsed_schema["security"]:
            futures.append(executor.submit(process_individual_part, 'security', security_item))
        # Handle footer
        futures.append(executor.submit(process_individual_part, 'tags', ""))

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                part_name, result = future.result()
                if result:
                    if part_name in ['paths', 'components']:
                        parts_dict[part_name].update(result)
                    else:
                        parts_dict[part_name] = result
                    yamlify_logger.info(f'Part processed successfully: {part_name}')
            except Exception as exc:
                yamlify_logger.info(f'Error with part: {exc}')

    # Merge the parts into the skeleton
    main_schema = merge_yaml('openapi_skelleton.yaml', parts_dict)

    # Load the merged YAML file
    # with open('merged_openapi_schema.yaml', 'r') as file:
    #     merged_yaml = yaml.safe_load(file)

    # Clean up the merged YAML
    cleaned_yaml = yaml_cleanup(main_schema)

    # Write the cleaned YAML to the final output file
    with open('final_openapi_schema.yaml', 'w') as file:
        yaml.safe_dump(cleaned_yaml, file, sort_keys=False)

    yamlify_logger.info("All parts have been merged and cleaned into the final OpenAPI schema.")


def generate_openapi_map() -> Dict:
    scrapped_content = test  # Example placeholder for scrapped content
    filename = "openapi_schema.json"
    max_length = 9000  # Define max length for each part of the documentation
    max_retries = 3  # Maximum retries for generating schema part

    parts = chunk_string(scrapped_content, max_length)
    total_parts = len(parts)
    openapi_schema = {}  # Start with an empty schema
    # saved_paths = {}  # Initialize saved_paths
    # saved_components = {"schemas": [], "securitySchemes": []}  # Initialize saved_components

    for i, part in enumerate(parts):
        part_number = i + 1
        part_tagged = f"==START OF PART {part_number}/{total_parts}==\n{part}\n==END OF PART {part_number}/{total_parts}==\n"
        yamlify_logger.info("\nProcessing part ", part_number, " of ", total_parts)
        yamlify_logger.info("\n===\nDocument sent to GPT-4o for processing\n===\n", part_tagged)

        # Extract paths and components related to the previous parts (1 to part_number-2)
        new_saved_paths = {}
        new_saved_components = {"schemas": [], "securitySchemes": []}

        for path, path_data in openapi_schema.get('paths', {}).items():
            for method_data in path_data.get('methods', []):
                if any(f"{j}/{total_parts}" in method_data.get('part', []) for j in range(1, part_number - 1)):
                    if path not in new_saved_paths:
                        new_saved_paths[path] = path_data

        for comp_key, comp_list in openapi_schema.get('components', {}).items():
            for comp in comp_list:
                if any(f"{j}/{total_parts}" in comp.get('part', []) for j in range(1, part_number - 1)):
                    new_saved_components[comp_key].append(comp)

        # Create a temporary schema without the saved paths and components
        openapi_schema_temp = json.loads(json.dumps(openapi_schema))  # Deep copy of the schema
        if 'paths' in openapi_schema_temp:
            for path in new_saved_paths:
                if path in openapi_schema_temp['paths']:
                    del openapi_schema_temp['paths'][path]

        if 'components' in openapi_schema_temp:
            for comp_key, comp_list in new_saved_components.items():
                for comp in comp_list:
                    if comp in openapi_schema_temp['components'].get(comp_key, []):
                        openapi_schema_temp['components'][comp_key].remove(comp)

        yamlify_logger.info("\n===\nSaved paths\n===\n", new_saved_paths)
        yamlify_logger.info("\n===\nSaved components\n===\n", new_saved_components)
        yamlify_logger.info("\n===\nOpenAPI TEMP Schema\n===\n", openapi_schema_temp)

        openapi_message = f"\n<Documentation>{part_tagged}\n</Documentation>\n<Schema>{json.dumps(openapi_schema_temp)}\n</Schema>"

        retry_count = 0
        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": systemMessageMapper},
                        {"role": "user", "content": humanMessage1},
                        {"role": "assistant", "content": aiMessage1},
                        {"role": "user", "content": openapi_message},
                    ]
                )

                output = response.choices[0].message.content
                schema_parts = yaml.safe_load(output)
                yamlify_logger.info("\n===\nSchema parts generated by GPT-4o\n===\n", schema_parts)

                # Convert paths from list to dictionary if needed
                if isinstance(schema_parts, dict) and 'paths' in schema_parts:
                    path_list = schema_parts['paths']
                    if isinstance(path_list, list):
                        schema_parts['paths'] = {item['path']: item for item in path_list}

                # Merge the new paths with saved paths
                for path, methods in schema_parts.get('paths', {}).items():
                    if path in new_saved_paths:
                        new_saved_paths[path].update(methods)
                    else:
                        new_saved_paths[path] = methods
                schema_parts['paths'] = new_saved_paths

                # Merge the new components with saved components
                for comp_key, comp_list in schema_parts.get('components', {}).items():
                    if comp_key not in new_saved_components:
                        new_saved_components[comp_key] = []
                    new_saved_components[comp_key].extend(comp_list)
                schema_parts['components'] = new_saved_components

                # Cleanup duplicates in schema parts
                schema_parts = cleanup_duplicates(schema_parts)

                # Merge schema parts
                for key, value in schema_parts.items():
                    if key == "paths":
                        if key in openapi_schema:
                            openapi_schema[key].update(value)
                        else:
                            openapi_schema[key] = value
                    elif key == "components":
                        if key in openapi_schema:
                            for comp_key, comp_value in value.items():
                                if comp_key in openapi_schema[key]:
                                    if isinstance(comp_value, list):
                                        openapi_schema[key][comp_key].extend(comp_value)
                                    elif isinstance(comp_value, dict):
                                        openapi_schema[key][comp_key].update(comp_value)
                                else:
                                    openapi_schema[key][comp_key] = comp_value
                        else:
                            openapi_schema[key] = value
                    elif key == "security":
                        if key in openapi_schema:
                            openapi_schema[key].extend(value)
                        else:
                            openapi_schema[key] = value
                    else:
                        openapi_schema[key] = value

                # Cleanup duplicates in the final schema
                openapi_schema = cleanup_duplicates(openapi_schema)

                yamlify_logger.info("\n===\nOpenAPI Schema updated\n===\n", openapi_schema)

                with open(filename, 'w') as file:
                    json.dump(openapi_schema, file, indent=4)
                yamlify_logger.info("\nUpdated OpenAPI Schema written to JSON file.\n")

                # # Update saved_paths and saved_components for the next iteration
                # saved_paths = new_saved_paths
                # saved_components = new_saved_components

                break  # Break the loop if processing is successful

            except yaml.YAMLError as e:
                retry_count += 1
                yamlify_logger.info(f"YAML parsing error on part {part_number}/{total_parts}: {e}")
                if retry_count < max_retries:
                    yamlify_logger.info(f"Retrying... ({retry_count}/{max_retries})")
                else:
                    yamlify_logger.info(
                        f"Failed to process part {part_number}/{total_parts} after {max_retries} retries. Moving to next part...")
            except Exception as e:
                yamlify_logger.info(f"Unexpected error: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    yamlify_logger.info(
                        f"Failed to process part {part_number}/{total_parts} after {max_retries} retries. Moving to next part...")

    return openapi_schema


def cleanup_duplicates(schema: Dict) -> Dict:
    """ Function to remove duplicates in lists of dictionaries based on deep equality """
    def remove_duplicates(lst):
        seen = []
        for item in lst:
            if not any(DeepDiff(item, seen_item, ignore_order=True) == {} for seen_item in seen):
                seen.append(item)
        return seen

    if 'components' in schema:
        for key, value in schema['components'].items():
            if isinstance(value, list):
                schema['components'][key] = remove_duplicates(value)

    if 'security' in schema:
        if isinstance(schema['security'], list):
            schema['security'] = remove_duplicates(schema['security'])

    return schema


def chunk_string(content: str, max_length: int) -> List[str]:
    """Splits text into chunks of `max_length` characters. If content length is less than `max_length` returns single chunk in the list"""
    total_length_content = len(content)

    if total_length_content <= max_length:
        yamlify_logger.debug(
            f"Length of content ({total_length_content=}) is less or equal than {max_length=} characters")
        return [content]

    yamlify_logger.debug(f"Length of content ({total_length_content=}) is greater than {max_length=} characters")
    chunks = []
    start = 0
    while start < total_length_content:
        # Search for a natural break point but do not exceed the bounds of the string
        if start + max_length >= total_length_content:
            next_break = total_length_content  # If the remaining content is less than max_length, take all of it
        else:
            next_break = content.rfind("\n\n", start, start + max_length)
            if next_break == -1:  # No double newline found, search for the last space within the limit
                next_break = content.rfind(" ", start, start + max_length)
                if next_break == -1:  # No space found either, force a break at max_length
                    next_break = start + max_length
        chunk = content[start:next_break].strip()
        chunks.append(chunk)
        start = next_break + 1  # Move start just past the last character of the current part

    yamlify_logger.debug(f"Content split into {len(chunks)} chunks")
    return chunks


def generate_openapi_part(document_parts: List, openapi_schema: Dict, part_name: str,
                          parsed_string: Union[str, Dict]) -> Dict:
    # Define part-specific conversation starters
    conversation_starters = {
        'header': "Need to write the full header of the openapi based on the Plan ({parsed_string}), please provide ALL information relevant for this API (description, etc.) that will go in the header",
        'paths': "\n FOCUS ONLY on this path please \n {parsed_string}",
        'components': "\n FOCUS ONLY on this component please \n {parsed_string}",
        'security': "\n FOCUS ONLY on this security please \n {parsed_string}",
        'footer': "Need to write relevant tag part based on the Plan if relevant (DO NOT invent any tag not present in the Plan)"
    }

    # Check the part name is valid
    if part_name not in conversation_starters:
        raise ValueError(f'Invalid part name: {part_name}. Must be one of {list(conversation_starters.keys())}.')

    # Ensure parsed_string is managed as a dictionary if provided as a JSON-formatted string
    if isinstance(parsed_string, str) and parsed_string:
        try:
            parsed_string_dict = json.loads(parsed_string)
        except json.JSONDecodeError:
            raise ValueError("parsed_string is not valid JSON.")
    elif isinstance(parsed_string, dict):
        parsed_string_dict = parsed_string
    else:
        if parsed_string:
            raise TypeError('parsed_string should be a valid JSON string or a dictionary.')
        else:
            parsed_string_dict = {}

    # Handle no content based on part
    if not parsed_string_dict and part_name != 'footer':
        return {}

    # Extract part indices and generate documentation stitch if part information is specified
    if "part" in parsed_string_dict:
        part_indices = [int(p.split('/')[0]) - 1 for p in parsed_string_dict["part"]]
    else:
        part_indices = []
    total_parts = len(document_parts)
    documentation_stitch = "\n".join(
        f"==START OF PART {idx + 1}/{total_parts}==\n{document_parts[idx]}\n==END OF PART {idx + 1}/{total_parts}=="
        for idx in part_indices
    ) if part_indices else ''

    # Use the defined conversation starter for the specified part
    if part_name != 'footer':
        conversation_starter = conversation_starters[part_name].format(
            parsed_string=json.dumps(parsed_string_dict, indent=2))
    else:
        conversation_starter = conversation_starters[part_name]

    openapi_message = f"\n<Plan>{json.dumps(openapi_schema)}\n</Plan>\n<Focus>{conversation_starter}\n</Focus>\n<Documentation>{documentation_stitch}\n</Documentation>"

    # Call the GPT API with the generated message
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": systemMessagePart},
            {"role": "user", "content": openapi_message}
        ]
    )
    output = response.choices[0].message.content
    yaml_openapi_part = yaml.safe_load(output)

    return yaml_openapi_part


def parse_openapi_schema(data: Dict) -> Dict:
    # Initialize separate dictionaries
    paths_dict = {}
    components_dict = {
        'schemas': [],
        'securitySchemes': []
    }
    security_dict = []

    # Extract title and servers
    title = data.get("title", "No title provided")
    servers = data.get("servers", [])

    # Process paths
    paths = data.get("paths", {})
    for path_key, path_value in paths.items():
        paths_dict[path_key] = {
            "methods": path_value.get("methods", [])
        }

    # Process components
    if "components" in data:
        if "schemas" in data["components"]:
            for schema in data["components"]["schemas"]:
                components_dict['schemas'].append({
                    "name": schema["name"],
                    "type": schema["type"],
                    "properties": schema.get("properties", []),
                    "part": schema.get("part", [])
                })

        if "securitySchemes" in data["components"]:
            for scheme in data["components"]["securitySchemes"]:
                components_dict['securitySchemes'].append({
                    "scheme": scheme["scheme"],
                    "part": scheme.get("part", [])
                })

    # Process security
    if "security" in data:
        for sec in data["security"]:
            security_dict.append({
                "securityRequirement": sec.get("securityRequirement", ""),
                "part": sec.get("part", [])
            })

    # Create a structured dictionary with the processed components
    processed_data = {
        "title": title,
        "servers": servers,
        "paths": paths_dict,
        "components": components_dict,
        "security": security_dict
    }

    return processed_data


def merge_yaml(skeleton_filename: str, parts_dict: Dict, output_name: str = 'openapi_schema.yaml') -> Dict:
    with open(skeleton_filename, 'r') as file:
        main_schema = yaml.safe_load(file) or {}

    def recursive_merge(dict1: dict, dict2: dict, path: str = ""):
        """
        Recursively merge two dictionaries.
        """
        for key, value in dict2.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                node = dict1.setdefault(key, {})
                if isinstance(node, dict):
                    recursive_merge(node, value, current_path)
                else:
                    dict1[key] = value
            elif isinstance(value, list):
                if key not in dict1:
                    dict1[key] = []
                if isinstance(dict1[key], list):
                    dict1[key].extend(value)
                else:
                    dict1[key] = value
            else:
                if key in dict1 and isinstance(dict1[key], (dict, list)):
                    # Skip merging due to type conflict
                    yamlify_logger.info(f"Skipping merge at key '{current_path}' due to type conflict.")
                else:
                    dict1[key] = value

    def cleanup_yaml(yaml_dict: Union[List, Dict]) -> Union[List, Dict]:
        """
        Recursively remove unwanted keys from the YAML dictionary.
        """
        if isinstance(yaml_dict, dict):
            cleaned_dict = {}
            for key, value in yaml_dict.items():
                if key != 'part':
                    cleaned_value = cleanup_yaml(value)
                    if cleaned_value or cleaned_value == 0:
                        cleaned_dict[key] = cleaned_value
            return cleaned_dict
        elif isinstance(yaml_dict, list):
            return [cleanup_yaml(item) for item in yaml_dict if cleanup_yaml(item) or cleanup_yaml(item) == 0]
        else:
            return yaml_dict

    # Ensure 'openapi' key only appears once at the top level
    if 'openapi' in parts_dict:
        main_schema['openapi'] = parts_dict['openapi']
        del parts_dict['openapi']

    # Flatten 'info' dictionary and remove unwanted keys
    if 'info' in parts_dict:
        info_part = parts_dict['info']
        if 'openapi' in info_part:
            del info_part['openapi']
        if 'components' in info_part:
            del info_part['components']
        if 'tags' in info_part:
            del info_part['tags']
        if 'security' in info_part:
            del info_part['security']
        if 'servers' in info_part:
            main_schema['servers'] = info_part['servers']
            del info_part['servers']
        main_schema['info'] = info_part

    for part, content in parts_dict.items():
        if content:
            part_data = content if isinstance(content, dict) else yaml.safe_load(content)
            if part == 'components' and 'info' in part_data:
                del part_data['info']
            recursive_merge(main_schema, {part: part_data})

    # Ensure components are not nested within info
    if 'components' in main_schema.get('info', {}):
        del main_schema['info']['components']

    # Final cleanup to remove duplicate nested categories
    def final_cleanup(schema):
        for key in ['info', 'components', 'tags', 'security']:
            if key in schema and isinstance(schema[key], dict) and key in schema[key]:
                schema[key] = schema[key][key]

    final_cleanup(main_schema)

    cleaned_schema = cleanup_yaml(main_schema)

    # Fix schema errors
    fixed_schema = fix_schema_errors(cleaned_schema)

    # Validate the fixed schema
    try:
        validate(fixed_schema, fixed_schema, cls=OAS30Validator)
        yamlify_logger.info("OpenAPI schema is valid.")
    except ValidationError as e:
        yamlify_logger.info(f"OpenAPI schema validation error: {e.message}")
        yamlify_logger.info("Could not automatically fix all schema errors.")

    with open(output_name, 'w') as file:
        yaml.safe_dump(fixed_schema, file, sort_keys=False)

    yamlify_logger.info(f"Merged and cleaned YAML written to {output_name}")
    return fixed_schema


def yaml_cleanup(yaml_dict: Union[List, Dict]) -> Union[List, Dict]:
    """
    Recursively remove empty parts from the YAML dictionary.
    """
    if isinstance(yaml_dict, dict):
        cleaned_dict = {}
        for key, value in yaml_dict.items():
            cleaned_value = yaml_cleanup(value)
            if cleaned_value or cleaned_value is False or cleaned_value == 0:
                cleaned_dict[key] = cleaned_value
        return cleaned_dict
    elif isinstance(yaml_dict, list):
        return [yaml_cleanup(item) for item in yaml_dict if
                yaml_cleanup(item) or yaml_cleanup(item) is False or yaml_cleanup(item) == 0]

    return yaml_dict


def fix_schema_errors(schema: Dict) -> Dict:
    """
    Attempt to fix common OpenAPI schema errors.
    """

    def fix_parameter(parameter: Dict):
        """
        Fix issues with individual parameters.
        """
        if parameter.get('in') == 'query' and 'default' in parameter:
            yamlify_logger.info(f"Removing 'default' from query parameter '{parameter['name']}'")
            del parameter['default']

        # Ensure parameters have either a 'schema' or 'content' property
        if 'schema' not in parameter and 'content' not in parameter:
            yamlify_logger.info(f"Adding 'schema' to parameter '{parameter['name']}'")
            parameter['schema'] = {}

    def fix_schema(schema: Dict):
        """
        Recursively fix issues in schema objects.
        """
        if 'properties' in schema:
            for prop_name, prop_value in schema['properties'].items():
                fix_schema(prop_value)

        if 'additionalProperties' in schema and isinstance(schema['additionalProperties'], dict):
            fix_schema(schema['additionalProperties'])

        if 'items' in schema:
            if isinstance(schema['items'], dict):
                fix_schema(schema['items'])
            elif isinstance(schema['items'], list):
                for item in schema['items']:
                    fix_schema(item)

        if 'required' in schema and not isinstance(schema['required'], list):
            yamlify_logger.info(f"Converting 'required' to list in schema {schema}")
            schema['required'] = [schema['required']]

        if 'enum' in schema and not isinstance(schema['enum'], list):
            yamlify_logger.info(f"Converting 'enum' to list in schema {schema}")
            schema['enum'] = [schema['enum']]

        if 'allOf' in schema and not isinstance(schema['allOf'], list):
            yamlify_logger.info(f"Converting 'allOf' to list in schema {schema}")
            schema['allOf'] = [schema['allOf']]

        if 'oneOf' in schema and not isinstance(schema['oneOf'], list):
            yamlify_logger.info(f"Converting 'oneOf' to list in schema {schema}")
            schema['oneOf'] = [schema['oneOf']]

        if 'anyOf' in schema and not isinstance(schema['anyOf'], list):
            yamlify_logger.info(f"Converting 'anyOf' to list in schema {schema}")
            schema['anyOf'] = [schema['anyOf']]

        if '$ref' in schema:
            ref = schema['$ref']
            if not ref.startswith('#/components/schemas/'):
                yamlify_logger.info(f"Removing invalid $ref: {ref}")
                del schema['$ref']

    def fix_responses(responses: Dict):
        """
        Fix issues in responses.
        """
        for response_code, response in responses.items():
            if 'content' in response:
                for content_type, content_value in response['content'].items():
                    if 'schema' in content_value:
                        fix_schema(content_value['schema'])

    def fix_paths(paths: Dict):
        """
        Fix issues in paths.
        """
        for path, path_item in paths.items():
            for method, method_item in path_item.items():
                if 'parameters' in method_item:
                    for parameter in method_item['parameters']:
                        fix_parameter(parameter)

                if 'requestBody' in method_item:
                    for content_type, content_value in method_item['requestBody'].get('content', {}).items():
                        if 'schema' in content_value:
                            fix_schema(content_value['schema'])

                if 'responses' in method_item:
                    fix_responses(method_item['responses'])

    # Fix paths
    if 'paths' in schema:
        fix_paths(schema['paths'])

    # Fix components
    if 'components' in schema:
        components = schema['components']
        if 'schemas' in components:
            if isinstance(components['schemas'], list):
                yamlify_logger.info("Converting 'schemas' from list to dictionary.")
                # Convert list to dictionary
                schemas_dict = {}
                for schema_item in components['schemas']:
                    schema_name = schema_item.get('name')
                    if schema_name:
                        schemas_dict[schema_name] = schema_item
                components['schemas'] = schemas_dict

            for schema_name, schema_value in components['schemas'].items():
                fix_schema(schema_value)

        if 'parameters' in components:
            for parameter_name, parameter_value in components['parameters'].items():
                fix_parameter(parameter_value)

        if 'responses' in components:
            fix_responses(components['responses'])

        if 'requestBodies' in components:
            for request_body_name, request_body_value in components['requestBodies'].items():
                for content_type, content_value in request_body_value.get('content', {}).items():
                    if 'schema' in content_value:
                        fix_schema(content_value['schema'])

    return schema


## try the generate_openapi_documentation function with a sample URL : https://finnhub.io/docs/api/

## yamlify_logger.info(generate_openapi_map("https://finnhub.io/docs/api/"))


# Example usage
# Assuming 'api_json' is the JSON structure you provided earlier

# generate_openapi_map("https://finnhub.io/docs/api/")
document_parts = chunk_string(test, 9000)

file_path = "openapi_schema.json"
with open(file_path, 'r') as file:
    openapi_schema = json.load(file)

parsed_schema = parse_openapi_schema(openapi_schema)

process_parts_concurrently(document_parts, openapi_schema, parsed_schema)

# #Test the Header
# part_name = 'header'
# parsed_schema_string = {"title": parsed_schema["title"], "servers": parsed_schema["servers"]}
# header_yaml_part = generate_openapi_part(document_parts, openapi_schema, part_name, parsed_schema_string)
# yamlify_logger.info(f"Generated Header YAML:\n{header_yaml_part}\n\n")

# # Test one Path
# part_name = 'paths'
# # Getting the first item from paths dictionary for testing
# path_key = next(iter(parsed_schema["paths"]))
# parsed_schema_string = {path_key: parsed_schema["paths"][path_key]}

# paths_yaml_part = generate_openapi_part(document_parts, openapi_schema, part_name, parsed_schema_string)
# yamlify_logger.info(f"Generated Path YAML for {path_key}:\n{paths_yaml_part}\n\n")

# # Test the Footer
# part_name = 'footer'
# parsed_schema_string = ""  
# footer_yaml_part = generate_openapi_part(document_parts, openapi_schema, part_name, parsed_schema_string)
# yamlify_logger.info(f"Generated Footer YAML:\n{footer_yaml_part}\n\n")

# parts_dict = {
#     "info": header_yaml_part,
#     "paths": paths_yaml_part,
#     "components": "",  
#     "security": "", 
#     "tags": footer_yaml_part
# }
# yamlify_logger.info("Parts dictionary:")
# yamlify_logger.info(parts_dict)
# # Merge yaml parts
# merge_yaml('openapi_skelleton.yaml', parts_dict, 'final_openapi_schema.yaml')
# merge_yaml('openapi_skelleton.yaml', [header_yaml_part, paths_yaml_part, footer_yaml_part], 'final_openapi_schema.yaml')

# process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
# yamlify_logger.info(json.dumps(parsed_schema, indent=4))
