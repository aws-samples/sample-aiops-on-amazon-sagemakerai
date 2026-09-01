import re
import ast
import json


def extract_prompt_response(trace):
    """Extract clean prompt and response text from a trace row.

    Handles nested dicts, stringified lists/dicts, chat message formats,
    and strips Qwen3 think tags.

    Args:
        trace: A pandas Series or dict-like object with 'request' and 'response' keys.

    Returns:
        Tuple of (prompt, response) as clean strings, or (None, None) if extraction fails.
    """
    inputs = trace.get("request")
    outputs = trace.get("response")
    if inputs is None or outputs is None:
        return None, None

    # Parse inputs
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except json.JSONDecodeError:
            try:
                inputs = ast.literal_eval(inputs)
            except (ValueError, SyntaxError):
                pass

    prompt = _extract_user_text(inputs)

    # Parse outputs
    if isinstance(outputs, str):
        try:
            outputs = json.loads(outputs)
        except json.JSONDecodeError:
            try:
                outputs = ast.literal_eval(outputs)
            except (ValueError, SyntaxError):
                pass

    response = _extract_response_text(outputs)

    # Clean up
    if prompt:
        prompt = prompt.replace(" /no_think", "").strip()
    if response:
        response = re.sub(r"<think>\s*</think>\s*", "", response).strip()

    return prompt, response


def _extract_user_text(obj):
    """Recursively extract user message text from various formats."""
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return _extract_user_text(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            parsed = ast.literal_eval(obj)
            return _extract_user_text(parsed)
        except (ValueError, SyntaxError):
            pass
        return obj

    if isinstance(obj, dict):
        for key in ["query", "prompt", "question", "text", "content", "input"]:
            if key in obj:
                return _extract_user_text(obj[key])
        if obj.get("role") == "user" and "content" in obj:
            return _extract_user_text(obj["content"])
        return str(obj)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                if item.get("role") == "user":
                    return _extract_user_text(item.get("content", ""))
                if "text" in item:
                    return item["text"]
        texts = [_extract_user_text(item) for item in obj]
        texts = [t for t in texts if t]
        return texts[0] if texts else str(obj)

    return str(obj)


def _extract_response_text(obj):
    """Extract response text from various formats."""
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return _extract_response_text(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        return obj

    if isinstance(obj, dict):
        for key in ["response", "output", "text", "content", "completion"]:
            if key in obj:
                val = obj[key]
                if isinstance(val, str):
                    return val
                return _extract_response_text(val)
        if "choices" in obj:
            choices = obj["choices"]
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                return msg.get("content", str(obj))
        return str(obj)

    if isinstance(obj, list):
        texts = [_extract_response_text(item) for item in obj]
        texts = [t for t in texts if t]
        return texts[0] if texts else str(obj)

    return str(obj)
