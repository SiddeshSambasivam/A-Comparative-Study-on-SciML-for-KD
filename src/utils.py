import json


def load_json(path: str) -> dict:
    """Loads a config file"""

    with open(path) as file:
        json_data = json.load(file)

    return json_data
