import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROMPTS_FILE = "prompt_info.json"

# JSON Schema definitions
LABELS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "labels_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of label names"
                }
            },
            "required": ["labels"]
        }
    }
}

CLASS_DESCRIPTIONS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "class_descriptions_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "descriptions": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": "Dictionary mapping class names to lists of descriptions"
                }
            },
            "required": ["descriptions"]
        }
    }
}

def generate_response(prompt, dataType, num_items):
    client = OpenAI(api_key=os.getenv("OPENAI_KEY_API"))

    if dataType == "labels":
        schema = LABELS_SCHEMA
        item_type = "class labels"
    else:
        schema = CLASS_DESCRIPTIONS_SCHEMA
        item_type = "text descriptions for every class label"

    system_message = f"""You are a helpful assistant. 
    You MUST output valid JSON that matches the provided JSON schema.
    Do not output anything outside the JSON object.
    Generate exactly {num_items} {item_type}. Your response must contain precisely {num_items} {item_type}. No more, no fewer."""

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        response_format=schema
    )
    
    # Parse the JSON output from the response
    return json.loads(response.choices[0].message.content)


def load_prompts(file_path=PROMPTS_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[Error] {file_path} not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"[Error] Expected a JSON object in {file_path}.")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"[Error] Invalid JSON format in {file_path}.") from e

def prepare_label_prompt(data, N, task=None, classArr=None, llmArr=None):
    test_example = data["test_example"].replace("#", str(N)) 
    if task:
        test_example = test_example.replace("@", task) 
    if classArr:
        classes_str = ", ".join(f"“{c}”" for c in classArr)
        test_example = test_example.replace("$", classes_str)
    if llmArr:
        classes_str = ", ".join(f"“{c}”" for c in llmArr)
        test_example = test_example.replace("&", classes_str)
    result = data["prompt"] + "\n" + test_example
    return result

def prepare_description_prompt(data, K, classArr):
    classes_str = ", ".join(f"“{c}”" for c in classArr)
    test_example = data["test_example"].replace("{}", classes_str).replace("#", str(K))
    result = data["prompt"] + "\n" + test_example
    return result
