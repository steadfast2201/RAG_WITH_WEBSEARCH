from pydantic import BaseModel
from config_settings import MODEL_NAME, PROMPT
from ollama import chat
from datetime import datetime
import json
from typing import List

class Queries(BaseModel):
    """Schema to validate the model output."""
    queries: List[str]

def extract_queries(query: str, model: str = MODEL_NAME) -> List[str]:
    """
    Generates a list of precise and optimized search queries from a user's input query.

    This function interacts with a local LLM (using the Ollama API) to create queries
    suited for both web search and local document search.

    Args:
        query (str): The original input query provided by the user.
        model (str, optional): The model name to use for query generation. Defaults to MODEL_NAME.

    Returns:
        List[str]: A list of optimized search queries.

    Raises:
        ValueError: If the model response is not in the expected format.
    """
    # Prepare the formatted prompt with today's date
    formatted_prompt = PROMPT.format(
        date=datetime.today().strftime('%Y-%m-%d'), 
        input_query=query
    )

    # Call the model to generate queries
    response = chat(
        model=model,
        messages=[{"role": "user", "content": formatted_prompt}],
        options={"temperature": 0.2},
        format=Queries.model_json_schema()
    )

    # Extract the 'content' field from the model response
    content = response.get('message', {}).get('content', None)

    if content is None:
        raise ValueError("No 'content' field found in the model response.")

    try:
        # Attempt to parse the content as JSON
        parsed_response = json.loads(content)
        queries = parsed_response.get("queries", None)

        if queries is None or not isinstance(queries, list):
            raise ValueError("Parsed response does not contain a valid 'queries' list.")

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model response as JSON: {e}")

    return queries