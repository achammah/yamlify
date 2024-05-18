import concurrent.futures
import json
import os
from typing import Dict, List, Tuple, Union

import yaml
from deepdiff import DeepDiff
from dotenv import load_dotenv
from jsonschema import ValidationError
from openai import OpenAI
from openapi_schema_validator import OAS30Validator, validate

from logger import Logger
from messageTemplate import aiMessage1, humanMessage1, systemMessageMapper, systemMessagePart, test


class OpenApiBuilder:
    MAX_THREAD_WORKERS = 20

    def __init__(self, log_level=Logger.INFO):
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        assert openai_key
        self.openai_client = OpenAI(api_key=openai_key)
        self.logger = Logger(name="yamlify", level=log_level)

    def process_parts_concurrently(
        self,
        document_parts: List[str],
        openapi_schema: Dict,
        parsed_schema: Union[List, Dict],
    ):
        """
        Processes documents concurrently via running Thread workers. Merges all outputs from worker threads to yaml
        and writes to a file
        Parameters:
            document_parts (list): List of strings representing a document part.
            openapi_schema (dict): Openapi schema dictionary (raw).
            parsed_schema (list | dict): Parsed openapi schema dictionary.

        Returns:
            None
        """
        parts_dict = {"info": "", "paths": {}, "components": {}, "security": "", "tags": ""}
        part_name_mapping = {
            "info": "header",
            "tags": "footer",
            "components": "components",
            "security": "security",
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_THREAD_WORKERS) as executor:
            futures = []
            # Handle header
            part_data = {"title": parsed_schema["title"], "servers": parsed_schema["servers"]}
            exc = executor.submit(
                self.process_individual_part, "info", part_name_mapping, part_data, document_parts, openapi_schema
            )
            futures.append(exc)

            # Handle paths
            for path_key in parsed_schema["paths"]:
                part_data = {path_key: parsed_schema["paths"][path_key]}
                exc = executor.submit(
                    self.process_individual_part, "paths", part_name_mapping, part_data, document_parts, openapi_schema
                )
                futures.append(exc)

            # Handle components
            for component_key in parsed_schema["components"]:
                part_data = {component_key: parsed_schema["components"][component_key]}
                exc = executor.submit(
                    self.process_individual_part,
                    "components",
                    part_name_mapping,
                    part_data,
                    document_parts,
                    openapi_schema,
                )
                futures.append(exc)

            # Handle security
            for security_item in parsed_schema["security"]:
                futures.append(
                    executor.submit(
                        self.process_individual_part,
                        "security",
                        part_name_mapping,
                        security_item,
                        document_parts,
                        openapi_schema,
                    )
                )

            # Handle footer
            futures.append(
                executor.submit(
                    self.process_individual_part, "tags", part_name_mapping, "", document_parts, openapi_schema
                )
            )

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    part_name, result = future.result()
                    if result:
                        if part_name in ["paths", "components"]:
                            parts_dict[part_name].update(result)
                        else:
                            parts_dict[part_name] = result
                        self.logger.info(f"Part processed successfully: {part_name}")
                except Exception as exc:
                    self.logger.exception(f"Error with part: {exc}")

        # Merge the parts into the skeleton
        main_schema = self.merge_yaml("openapi_skeleton.yaml", parts_dict)

        # Load the merged YAML file
        # with open('merged_openapi_schema.yaml', 'r') as file:
        #     merged_yaml = yaml.safe_load(file)

        # Clean up the merged YAML
        cleaned_yaml = self.yaml_cleanup(main_schema)

        # Write the cleaned YAML to the final output file
        with open("final_openapi_schema.yaml", "w", encoding="UTF-8") as file:
            yaml.safe_dump(cleaned_yaml, file, sort_keys=False)

        self.logger.info("All parts have been merged and cleaned into the final OpenAPI schema.")

    def process_individual_part(
        self,
        part_name: str,
        part_name_mapping: Dict[str, str],
        part_data: Union[str, Dict],
        document_parts: List[str],
        openapi_schema: Dict,
    ) -> Union[Tuple[None, None], Tuple[str, Dict]]:
        """
        Parameters:
            part_name (str): The name of the part
            part_name_mapping (dict): Openapi part name mapping
            part_data (str | dict): part_data is expected to be a dict or string dict with details necessary for
                                    processing the part
            document_parts (list): List of strings representing a document part.
            openapi_schema (dict): Openapi schema dictionary (raw).

        Returns:
            openapi_part (tuple): part_name and dict containing openapi part. Returns None if there was an error.
        """
        try:
            mapped_part_name = part_name_mapping.get(part_name, part_name)
            self.logger.info(f"Processing part: {mapped_part_name}")
            openapi_part = part_name, self.generate_openapi_part(
                document_parts, openapi_schema, mapped_part_name, part_data
            )
            return openapi_part
        except Exception as exc:
            self.logger.exception(f"Error processing part: {exc}")
            return None, None

    def generate_openapi_map(self, output_filename: str = None, max_length: int = 9000, max_retries: int = 3) -> Dict:
        """
        Generates OpenAPI schema
        Parameters:
            output_filename (str): Output filename, defaults to "openapi_schema.json"
            max_length (dict): Define max length for each part of the documentation
            max_retries (str | dict): Maximum retries for generating schema part

        Returns:
            openapi_schema (dict): openapi schema
        """
        scrapped_content = test  # Example placeholder for scrapped content
        if not output_filename:
            output_filename = "openapi_schema.json"

        parts = self.chunk_string(scrapped_content, max_length)
        total_parts = len(parts)
        openapi_schema = {}  # Start with an empty schema
        # saved_paths = {}  # Initialize saved_paths
        # saved_components = {"schemas": [], "securitySchemes": []}  # Initialize saved_components

        for i, part in enumerate(parts):
            part_number = i + 1
            part_tagged = (
                f"==START OF PART {part_number}/{total_parts}==\n{part}\n==END OF PART {part_number}/{total_parts}==\n"
            )
            self.logger.debug(f"\nProcessing part {part_number} of {total_parts}")
            self.logger.debug(
                f"\n===\nDocument sent to GPT-4o for processing\n===\n{part_tagged}",
            )

            # Extract paths and components related to the previous parts (1 to part_number-2)
            new_saved_paths = {}
            new_saved_components = {"schemas": [], "securitySchemes": []}

            self.populate_paths(openapi_schema, new_saved_paths, total_parts, part_number)
            self.populate_components(openapi_schema, new_saved_components, total_parts, part_number)

            # Create a temporary schema without the saved paths and components
            openapi_schema_temp = json.loads(json.dumps(openapi_schema))  # Deep copy of the schema
            if "paths" in openapi_schema_temp:
                for path in new_saved_paths:
                    if path in openapi_schema_temp["paths"]:
                        del openapi_schema_temp["paths"][path]

            if "components" in openapi_schema_temp:
                for comp_key, comp_list in new_saved_components.items():
                    for comp in comp_list:
                        if comp in openapi_schema_temp["components"].get(comp_key, []):
                            openapi_schema_temp["components"][comp_key].remove(comp)

            self.logger.debug(f"\n===\nSaved paths\n===\n{new_saved_paths}")
            self.logger.debug(f"\n===\nSaved components\n===\n{new_saved_components}")
            self.logger.debug(f"\n===\nOpenAPI TEMP Schema\n===\n{openapi_schema_temp}")

            openapi_message = f"\n<Documentation>{part_tagged}\n</Documentation>\n<Schema>{json.dumps(openapi_schema_temp)}\n</Schema>"

            retry_count = 0
            is_success = False
            while retry_count < max_retries and not is_success:
                try:
                    is_success = self.generate_openai_response(
                        openapi_message, new_saved_paths, new_saved_components, openapi_schema, output_filename
                    )
                except yaml.YAMLError as e:
                    retry_count += 1
                    self.logger.info(f"YAML parsing error on part {part_number}/{total_parts}: {e}")
                    if retry_count < max_retries:
                        self.logger.info(f"Retrying... ({retry_count}/{max_retries})")
                    else:
                        self.logger.error(
                            f"Failed to process part {part_number}/{total_parts} after {max_retries} retries. Moving to next part..."
                        )
                except Exception as e:
                    self.logger.exception(f"Unexpected error: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        self.logger.error(
                            f"Failed to process part {part_number}/{total_parts} after {max_retries} retries. Moving to next part..."
                        )

        return openapi_schema

    def populate_components(self, openapi_schema: Dict, new_saved_components: Dict, total_parts: int, part_number: int):
        """
        Populates paths from openapi_schema and saves them to new_saved_paths.
        Parameters:
            openapi_schema (dict): Openapi schema
            new_saved_components (dict): Dict containing components. New components will be added to this dict
            total_parts (dict): Total parts in openapi_schema
            part_number (dict): Part number in openapi_schema

        Returns:
            None
        """
        for comp_key, comp_list in openapi_schema.get("components", {}).items():
            for comp in comp_list:
                if any(f"{j}/{total_parts}" in comp.get("part", []) for j in range(1, part_number - 1)):
                    new_saved_components[comp_key].append(comp)

    def populate_paths(self, openapi_schema: Dict, new_saved_paths: Dict, total_parts: int, part_number: int):
        """
        Populates paths from openapi_schema and saves them to new_saved_paths.
        Parameters:
            openapi_schema (dict): Openapi schema
            new_saved_paths (dict): Dict containing paths. New paths will be added to this dict
            total_parts (dict): Total parts in openapi_schema
            part_number (dict): Part number in openapi_schema

        Returns:
            None
        """
        for path, path_data in openapi_schema.get("paths", {}).items():
            for method_data in path_data.get("methods", []):
                if any(f"{j}/{total_parts}" in method_data.get("part", []) for j in range(1, part_number - 1)):
                    if path not in new_saved_paths:
                        new_saved_paths[path] = path_data

    def generate_openai_response(
        self,
        openapi_message: str,
        new_saved_paths: dict,
        new_saved_components: dict,
        openapi_schema: dict,
        output_filename: str,
    ) -> bool:
        """
        Generates OpenAPI schema via utilizing openai GPT-o api and writes output to a file
        Returns True if processing is successful.
        Raises exception on error
        Parameters:
            openapi_message (str): String containing documentation for openapi input
            new_saved_paths (dict): Dict containing paths. New paths will be added to this dict
            new_saved_components (dict): Dict containing openapi components
            openapi_schema (dict): Dict containing openapi schema. Schema will be updated
            output_filename (str): Output filename, defaults to "openapi_schema.json"

        Returns:
            is_success (bool): Operation success result
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": systemMessageMapper},
                {"role": "user", "content": humanMessage1},
                {"role": "assistant", "content": aiMessage1},
                {"role": "user", "content": openapi_message},
            ],
        )

        output = response.choices[0].message.content
        schema_parts = yaml.safe_load(output)
        self.logger.debug(f"\n===\nSchema parts generated by GPT-4o\n===\n{schema_parts}")

        # Convert paths from list to dictionary if needed
        if isinstance(schema_parts, dict) and "paths" in schema_parts:
            path_list = schema_parts["paths"]
            if isinstance(path_list, list):
                schema_parts["paths"] = {item["path"]: item for item in path_list}

        # Merge the new paths with saved paths
        for path, methods in schema_parts.get("paths", {}).items():
            if path in new_saved_paths:
                new_saved_paths[path].update(methods)
            else:
                new_saved_paths[path] = methods
        schema_parts["paths"] = new_saved_paths

        # Merge the new components with saved components
        for comp_key, comp_list in schema_parts.get("components", {}).items():
            if comp_key not in new_saved_components:
                new_saved_components[comp_key] = []
            new_saved_components[comp_key].extend(comp_list)
        schema_parts["components"] = new_saved_components

        # Cleanup duplicates in schema parts
        schema_parts = self.cleanup_duplicates(schema_parts)

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
        openapi_schema = self.cleanup_duplicates(openapi_schema)

        self.logger.debug(f"\n===\nOpenAPI Schema updated\n===\n{openapi_schema}")

        with open(output_filename, "w", encoding="UTF-8") as file:
            json.dump(openapi_schema, file, indent=4)
        self.logger.info("\nUpdated OpenAPI Schema written to JSON file.\n")

        # # Update saved_paths and saved_components for the next iteration
        # saved_paths = new_saved_paths
        # saved_components = new_saved_components

        # processing successful
        return True

    def remove_duplicates(self, lst: List):
        """
        Function to remove duplicates in the list
        Parameters:
            lst (list): List to be cleaned from duplicate items

        Returns:
            clean_lst (list): Cleaned list
        """
        clean_lst = []
        for item in lst:
            if not any(DeepDiff(item, seen_item, ignore_order=True) == {} for seen_item in clean_lst):
                clean_lst.append(item)

        self.logger.debug("Removed duplicates from list")
        return clean_lst

    def cleanup_duplicates(self, schema: Dict) -> Dict:
        """
        Function to remove duplicates in lists of dictionaries based on deep equality
        Parameters:
            schema (dict): Dict containing openapi schema.

        Returns:
            schema (dict): Cleaned schema
        """
        if "components" in schema:
            for key, value in schema["components"].items():
                if isinstance(value, list):
                    schema["components"][key] = self.remove_duplicates(value)

        if "security" in schema:
            if isinstance(schema["security"], list):
                schema["security"] = self.remove_duplicates(schema["security"])

        return schema

    def chunk_string(self, content: str, max_length: int) -> List[str]:
        """
        Splits text into chunks of `max_length` characters. If content length is less than `max_length` returns single
        chunk in the list
        Parameters:
            content (str): content to be split
            max_length (int): max length of each chunk

        Returns:
            chunks (list): list of chunks
        """
        total_length_content = len(content)

        if total_length_content <= max_length:
            self.logger.debug(
                f"Length of content ({total_length_content=}) is less or equal than {max_length=} characters"
            )
            return [content]

        self.logger.debug(f"Length of content ({total_length_content=}) is greater than {max_length=} characters")
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

        self.logger.debug(f"Content split into {len(chunks)} chunks")
        return chunks

    def generate_openapi_part(
        self, document_parts: List, openapi_schema: Dict, part_name: str, parsed_string: Union[str, Dict]
    ) -> Dict:
        """
        Generates openapi part via utilizing openai GPT-o api
        Parameters:
            document_parts (list): List of document parts
            openapi_schema (dict): Openapi schema
            part_name (str): Openapi schema part name
            parsed_string (str | dict): Parsed string

        Returns:
            yaml_openapi_part (dict): yaml doc of generated openapi part
        """
        # Define part-specific conversation starters
        conversation_starters = {
            "header": "Need to write the full header of the openapi based on the Plan ({parsed_string}), please provide ALL information relevant for this API (description, etc.) that will go in the header",
            "paths": "\n FOCUS ONLY on this path please \n {parsed_string}",
            "components": "\n FOCUS ONLY on this component please \n {parsed_string}",
            "security": "\n FOCUS ONLY on this security please \n {parsed_string}",
            "footer": "Need to write relevant tag part based on the Plan if relevant (DO NOT invent any tag not present in the Plan)",
        }

        # Check the part name is valid
        if part_name not in conversation_starters:
            raise ValueError(f"Invalid part name: {part_name}. Must be one of {list(conversation_starters.keys())}.")

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
                raise TypeError("parsed_string should be a valid JSON string or a dictionary.")
            else:
                parsed_string_dict = {}

        # Handle no content based on part
        if not parsed_string_dict and part_name != "footer":
            return {}

        # Extract part indices and generate documentation stitch if part information is specified
        if "part" in parsed_string_dict:
            part_indices = [int(p.split("/")[0]) - 1 for p in parsed_string_dict["part"]]
        else:
            part_indices = []
        total_parts = len(document_parts)
        documentation_stitch = (
            "\n".join(
                f"==START OF PART {idx + 1}/{total_parts}==\n{document_parts[idx]}\n==END OF PART {idx + 1}/{total_parts}=="
                for idx in part_indices
            )
            if part_indices
            else ""
        )

        # Use the defined conversation starter for the specified part
        if part_name != "footer":
            conversation_starter = conversation_starters[part_name].format(
                parsed_string=json.dumps(parsed_string_dict, indent=2)
            )
        else:
            conversation_starter = conversation_starters[part_name]

        openapi_message = f"\n<Plan>{json.dumps(openapi_schema)}\n</Plan>\n<Focus>{conversation_starter}\n</Focus>\n<Documentation>{documentation_stitch}\n</Documentation>"

        # Call the GPT API with the generated message
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": systemMessagePart}, {"role": "user", "content": openapi_message}],
        )
        output = response.choices[0].message.content
        yaml_openapi_part = yaml.safe_load(output)

        return yaml_openapi_part

    def parse_openapi_schema(self, data: Dict) -> Dict:
        """
        Parses openapi schema
        Parameters:
            data (dict): Openapi schema

        Returns:
            processed_data (dict): dict containing parsed openapi schema from input
        """
        # Initialize separate dictionaries
        paths_dict = {}
        components_dict = {"schemas": [], "securitySchemes": []}
        security_dict = []

        # Extract title and servers
        title = data.get("title", "No title provided")
        servers = data.get("servers", [])

        # Process paths
        paths = data.get("paths", {})
        for path_key, path_value in paths.items():
            paths_dict[path_key] = {"methods": path_value.get("methods", [])}

        # Process components
        if "components" in data:
            if "schemas" in data["components"]:
                for schema in data["components"]["schemas"]:
                    components_dict["schemas"].append(
                        {
                            "name": schema["name"],
                            "type": schema["type"],
                            "properties": schema.get("properties", []),
                            "part": schema.get("part", []),
                        }
                    )

            if "securitySchemes" in data["components"]:
                for scheme in data["components"]["securitySchemes"]:
                    components_dict["securitySchemes"].append(
                        {"scheme": scheme["scheme"], "part": scheme.get("part", [])}
                    )

        # Process security
        if "security" in data:
            for sec in data["security"]:
                security_dict.append(
                    {"securityRequirement": sec.get("securityRequirement", ""), "part": sec.get("part", [])}
                )

        # Create a structured dictionary with the processed components
        processed_data = {
            "title": title,
            "servers": servers,
            "paths": paths_dict,
            "components": components_dict,
            "security": security_dict,
        }
        self.logger.debug("Parsed openapi schema")

        return processed_data

    def recursive_merge(self, dict1: dict, dict2: dict, path: str = ""):
        """
        Recursively merge two dictionaries.
        Parameters:
            dict1 (dict): First dictionary
            dict2 (dict): Second dictionary will be merged to dict1
            path (str): openapi part name

        Returns:
            None
        """
        for key, value in dict2.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                node = dict1.setdefault(key, {})
                if isinstance(node, dict):
                    self.recursive_merge(node, value, current_path)
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
                    self.logger.info(f"Skipping merge at key '{current_path}' due to type conflict.")
                else:
                    dict1[key] = value

    def cleanup_yaml(self, yaml_dict: Union[List, Dict]) -> Union[List, Dict]:
        """
        Recursively remove unwanted keys from the YAML dictionary.
        Parameters:
            yaml_dict (list | dict): yaml dictionary to be cleaned

        Returns:
            yaml_dict | cleaned_dict (dict): cleaned dictionary
        """
        if isinstance(yaml_dict, dict):
            cleaned_dict = {}
            for key, value in yaml_dict.items():
                if key != "part":
                    cleaned_value = self.cleanup_yaml(value)
                    if cleaned_value or cleaned_value == 0:
                        cleaned_dict[key] = cleaned_value
            return cleaned_dict
        elif isinstance(yaml_dict, list):
            clean_list = []
            for item in yaml_dict:
                cleaned_item = self.cleanup_yaml(item)
                if self.cleanup_yaml(item) == 0:
                    clean_list.append(cleaned_item)

            return clean_list
        else:
            return yaml_dict

    def merge_yaml(self, skeleton_filename: str, parts_dict: Dict, output_name: str = None) -> Dict:
        """
        Merge yaml dictionaries into a single dictionary. and save it to a file.
        Parameters:
            skeleton_filename (str): Openapi yaml skeleton filename
            parts_dict (dict): Openapi parts dict
            output_name (str): Output file name

        Returns:
            fixed_schema (dict): Fixed openapi schema
        """
        if not output_name:
            output_name = "openapi_schema.yaml"

        with open(skeleton_filename, "r", encoding="UTF-8") as file:
            main_schema = yaml.safe_load(file) or {}

        # Ensure 'openapi' key only appears once at the top level
        if "openapi" in parts_dict:
            main_schema["openapi"] = parts_dict["openapi"]
            del parts_dict["openapi"]

        # Flatten 'info' dictionary and remove unwanted keys
        if "info" in parts_dict:
            info_part = parts_dict["info"]
            if "openapi" in info_part:
                del info_part["openapi"]
            if "components" in info_part:
                del info_part["components"]
            if "tags" in info_part:
                del info_part["tags"]
            if "security" in info_part:
                del info_part["security"]
            if "servers" in info_part:
                main_schema["servers"] = info_part["servers"]
                del info_part["servers"]
            main_schema["info"] = info_part

        for part, content in parts_dict.items():
            if content:
                part_data = content if isinstance(content, dict) else yaml.safe_load(content)
                if part == "components" and "info" in part_data:
                    del part_data["info"]
                self.recursive_merge(main_schema, {part: part_data})

        # Ensure components are not nested within info
        if "components" in main_schema.get("info", {}):
            del main_schema["info"]["components"]

        # Final cleanup to remove duplicate nested categories
        def final_cleanup(schema):
            for key in ["info", "components", "tags", "security"]:
                if key in schema and isinstance(schema[key], dict) and key in schema[key]:
                    schema[key] = schema[key][key]

        final_cleanup(main_schema)

        cleaned_schema = self.cleanup_yaml(main_schema)

        # Fix schema errors
        fixed_schema = self.fix_schema_errors(cleaned_schema)

        # Validate the fixed schema
        try:
            validate(fixed_schema, fixed_schema, cls=OAS30Validator)
            self.logger.info("OpenAPI schema is valid.")
        except ValidationError as e:
            self.logger.exception(f"OpenAPI schema validation error: {e.message}")
            self.logger.error("Could not automatically fix all schema errors.")

        with open(output_name, "w", encoding="UTF-8") as file:
            yaml.safe_dump(fixed_schema, file, sort_keys=False)

        self.logger.info(f"Merged and cleaned YAML written to {output_name}")
        return fixed_schema

    def yaml_cleanup(self, yaml_dict: Union[List, Dict]) -> Union[List, Dict]:
        """
        Recursively remove empty parts from the YAML dictionary.
        Might return the input back if the input type is not list or dict
        Parameters:
            yaml_dict (list | dict): Yaml dict or list to be cleaned

        Returns:
            yaml_dict (list | dict): cleaned yaml
        """
        if isinstance(yaml_dict, dict):
            cleaned_dict = {}
            for key, value in yaml_dict.items():
                cleaned_value = self.yaml_cleanup(value)
                if cleaned_value or cleaned_value is False or cleaned_value == 0:
                    cleaned_dict[key] = cleaned_value
            return cleaned_dict
        elif isinstance(yaml_dict, list):
            clean_list = []
            for item in yaml_dict:
                cleaned_item = self.yaml_cleanup(item)
                if cleaned_item or cleaned_item in [False, 0]:
                    clean_list.append(cleaned_item)

            return clean_list

        return yaml_dict

    def fix_schema_errors(self, schema: Dict) -> Dict:
        """
        Attempt to fix common OpenAPI schema errors.
        Parameters:
            schema (dict): Dict to be fixed for common openapi schema errors

        Returns:
            schema (dict): fixed schema
        """

        # Fix paths
        if "paths" in schema:
            self.fix_paths(schema["paths"])

        # Fix components
        if "components" in schema:
            components = schema["components"]
            if "schemas" in components:
                if isinstance(components["schemas"], list):
                    self.logger.info("Converting 'schemas' from list to dictionary.")
                    # Convert list to dictionary
                    schemas_dict = {}
                    for schema_item in components["schemas"]:
                        schema_name = schema_item.get("name")
                        if schema_name:
                            schemas_dict[schema_name] = schema_item
                    components["schemas"] = schemas_dict

                for schema_name, schema_value in components["schemas"].items():
                    self.fix_schema(schema_value)

            if "parameters" in components:
                for _, parameter_value in components["parameters"].items():
                    self.fix_parameter(parameter_value)

            if "responses" in components:
                self.fix_responses(components["responses"])

            if "requestBodies" in components:
                for _, request_body_value in components["requestBodies"].items():
                    for _, content_value in request_body_value.get("content", {}).items():
                        if "schema" in content_value:
                            self.fix_schema(content_value["schema"])

        return schema

    def fix_parameter(self, parameter: Dict):
        """
        Fix issues with individual parameters.
        Parameters:
            parameter (dict): Dict containing openapi parameters

        Returns:
            schema (dict): fixed openapi schema parameters
        """
        if parameter.get("in") == "query" and "default" in parameter:
            self.logger.info(f"Removing 'default' from query parameter '{parameter['name']}'")
            del parameter["default"]

        # Ensure parameters have either a 'schema' or 'content' property
        if "schema" not in parameter and "content" not in parameter:
            self.logger.info(f"Adding 'schema' to parameter '{parameter['name']}'")
            parameter["schema"] = {}

    def fix_schema(self, schema: Dict):
        """
        Recursively fix issues in schema objects.
        Parameters:
            schema (dict): Dict containing openapi schema to be fixed

        Returns:
            None
        """
        if "properties" in schema:
            for _, prop_value in schema["properties"].items():
                self.fix_schema(prop_value)

        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            self.fix_schema(schema["additionalProperties"])

        if "items" in schema:
            if isinstance(schema["items"], dict):
                self.fix_schema(schema["items"])
            elif isinstance(schema["items"], list):
                for item in schema["items"]:
                    self.fix_schema(item)

        if "required" in schema and not isinstance(schema["required"], list):
            self.logger.info(f"Converting 'required' to list in schema {schema}")
            schema["required"] = [schema["required"]]

        if "enum" in schema and not isinstance(schema["enum"], list):
            self.logger.info(f"Converting 'enum' to list in schema {schema}")
            schema["enum"] = [schema["enum"]]

        if "allOf" in schema and not isinstance(schema["allOf"], list):
            self.logger.info(f"Converting 'allOf' to list in schema {schema}")
            schema["allOf"] = [schema["allOf"]]

        if "oneOf" in schema and not isinstance(schema["oneOf"], list):
            self.logger.info(f"Converting 'oneOf' to list in schema {schema}")
            schema["oneOf"] = [schema["oneOf"]]

        if "anyOf" in schema and not isinstance(schema["anyOf"], list):
            self.logger.info(f"Converting 'anyOf' to list in schema {schema}")
            schema["anyOf"] = [schema["anyOf"]]

        if "$ref" in schema:
            ref = schema["$ref"]
            if not ref.startswith("#/components/schemas/"):
                self.logger.info(f"Removing invalid $ref: {ref}")
                del schema["$ref"]

    def fix_responses(self, responses: Dict):
        """
        Fix issues in responses.
        Parameters:
            responses (dict): Dict containing openapi responses to be fixed

        Returns:
            None
        """
        for _, response in responses.items():
            if "content" in response:
                for _, content_value in response["content"].items():
                    if "schema" in content_value:
                        self.fix_schema(content_value["schema"])

    def fix_paths(self, paths: Dict):
        """
        Fix issues in paths.
        Parameters:
            paths (dict): Dict containing openapi paths to be fixed

        Returns:
            None
        """
        for _, path_item in paths.items():
            for _, method_item in path_item.items():
                if "parameters" in method_item:
                    for parameter in method_item["parameters"]:
                        self.fix_parameter(parameter)

                if "requestBody" in method_item:
                    for _, content_value in method_item["requestBody"].get("content", {}).items():
                        if "schema" in content_value:
                            self.fix_schema(content_value["schema"])

                if "responses" in method_item:
                    self.fix_responses(method_item["responses"])


if __name__ == "__main__":
    openapi_schema_builder = OpenApiBuilder(Logger.DEBUG)
    doc_parts = openapi_schema_builder.chunk_string(test, 9000)

    file_path = "openapi_schema.json"
    with open(file_path, "r", encoding="UTF-8") as file:
        loaded_openapi_schema = json.load(file)

    parsed_openapi_schema = openapi_schema_builder.parse_openapi_schema(loaded_openapi_schema)

    openapi_schema_builder.process_parts_concurrently(doc_parts, loaded_openapi_schema, parsed_openapi_schema)

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
# merge_yaml('openapi_skeleton.yaml', parts_dict, 'final_openapi_schema.yaml')
# merge_yaml('openapi_skelleton.yaml', [header_yaml_part, paths_yaml_part, footer_yaml_part], 'final_openapi_schema.yaml')

# process_parts_concurrently(document_parts, openapi_schema, parsed_schema)
# yamlify_logger.info(json.dumps(parsed_schema, indent=4))
