"""Utility functions for working with MessageHistory and multi-modal content.

This module provides helper functions for creating messages with images, files,
and other non-text content in the standard OpenAI-compatible format.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import base64
import mimetypes


def create_text_content(text: str) -> Dict[str, str]:
    """Create a text content part.

    Args:
        text: The text content

    Returns:
        Content part dictionary
    """
    return {"type": "text", "text": text}


def create_image_url_content(url: str, detail: str = "auto") -> Dict[str, Any]:
    """Create an image URL content part.

    Args:
        url: The image URL
        detail: Detail level for vision models ("low", "high", or "auto")

    Returns:
        Content part dictionary
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": url,
            "detail": detail,
        },
    }


def create_image_file_content(file_path: Union[str, Path], detail: str = "auto", encode_base64: bool = False) -> Dict[str, Any]:
    """Create an image file content part.

    Args:
        file_path: Path to the image file
        detail: Detail level for vision models ("low", "high", or "auto")
        encode_base64: If True, encode image as base64 data URL

    Returns:
        Content part dictionary
    """
    file_path = Path(file_path)

    if encode_base64:
        # Read and encode as base64 data URL
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Detect MIME type
        mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"

        url = f"data:{mime_type};base64,{image_data}"
    else:
        # Use file path as URL
        url = f"file://{file_path.absolute()}"

    return {
        "type": "image_url",
        "image_url": {
            "url": url,
            "detail": detail,
        },
    }


def create_image_from_pil(pil_image: Any, detail: str = "auto", format: str = "PNG") -> Dict[str, Any]:
    """Create an image content part from a PIL Image object.

    This is a convenience function that converts PIL images to base64 data URLs.
    Note: Requires PIL/Pillow to be installed.

    Args:
        pil_image: A PIL.Image.Image object
        detail: Detail level for vision models ("low", "high", or "auto")
        format: Image format for encoding (PNG, JPEG, etc.)

    Returns:
        Content part dictionary with base64-encoded image

    Example:
        ```python
        from PIL import Image
        from maseval.core.utils.message_utils import create_image_from_pil

        img = Image.open("photo.jpg")
        content = create_image_from_pil(img, detail="high")
        ```
    """
    try:
        from io import BytesIO
    except ImportError as e:
        raise ImportError("io module not available") from e

    # Convert PIL image to base64
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Determine MIME type from format
    mime_types = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "GIF": "image/gif",
        "WEBP": "image/webp",
    }
    mime_type = mime_types.get(format.upper(), "image/png")

    url = f"data:{mime_type};base64,{image_data}"

    return {
        "type": "image_url",
        "image_url": {
            "url": url,
            "detail": detail,
        },
    }


def create_image_from_numpy(numpy_array: Any, detail: str = "auto", format: str = "PNG") -> Dict[str, Any]:
    """Create an image content part from a numpy array.

    This is a convenience function that converts numpy arrays to base64 data URLs.
    Note: Requires both numpy and PIL/Pillow to be installed.

    Args:
        numpy_array: A numpy array representing an image (H, W, C)
        detail: Detail level for vision models ("low", "high", or "auto")
        format: Image format for encoding (PNG, JPEG, etc.)

    Returns:
        Content part dictionary with base64-encoded image

    Example:
        ```python
        import numpy as np
        from maseval.core.utils.message_utils import create_image_from_numpy

        # Create or load a numpy array
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        content = create_image_from_numpy(img_array)
        ```
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("PIL/Pillow is required for numpy array conversion. Install with: pip install Pillow") from e

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(numpy_array)

    # Use the PIL conversion function
    return create_image_from_pil(pil_image, detail=detail, format=format)


def create_file_content(file_path: Union[str, Path], mime_type: Optional[str] = None) -> Dict[str, Any]:
    """Create a file content part.

    Args:
        file_path: Path to the file
        mime_type: MIME type of the file (auto-detected if not provided)

    Returns:
        Content part dictionary
    """
    file_path = Path(file_path)

    if mime_type is None:
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

    return {
        "type": "file",
        "file_path": str(file_path.absolute()),
        "mime_type": mime_type,
    }


def create_audio_content(file_path: Union[str, Path], mime_type: Optional[str] = None) -> Dict[str, Any]:
    """Create an audio content part.

    Args:
        file_path: Path to the audio file
        mime_type: MIME type of the audio (auto-detected if not provided)

    Returns:
        Content part dictionary
    """
    file_path = Path(file_path)

    if mime_type is None:
        mime_type = mimetypes.guess_type(file_path)[0] or "audio/mpeg"

    return {
        "type": "audio",
        "file_path": str(file_path.absolute()),
        "mime_type": mime_type,
    }


def create_multimodal_message(role: str, parts: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Create a multi-modal message with mixed content types.

    Args:
        role: The message role ("user", "assistant", "system")
        parts: List of content parts. Can be strings (converted to text) or dicts

    Returns:
        Message dictionary

    Example:
        ```python
        message = create_multimodal_message(
            "user",
            [
                "What's in this image?",
                create_image_url_content("https://example.com/image.jpg"),
                "And this one?",
                create_image_file_content("local_image.jpg")
            ]
        )
        ```
    """
    content = []
    for part in parts:
        if isinstance(part, str):
            content.append(create_text_content(part))
        else:
            content.append(part)

    return {
        "role": role,
        "content": content,
    }


def create_tool_call(tool_name: str, arguments: Union[str, Dict[str, Any]], call_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a tool call structure.

    Args:
        tool_name: Name of the tool/function to call
        arguments: Arguments as JSON string or dict
        call_id: Optional unique ID for the call (auto-generated if not provided)

    Returns:
        Tool call dictionary

    Example:
        ```python
        tool_call = create_tool_call(
            "get_weather",
            {"city": "New York", "units": "fahrenheit"}
        )
        ```
    """
    import json
    import uuid

    if call_id is None:
        call_id = f"call_{uuid.uuid4().hex[:24]}"

    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": arguments,
        },
    }


def extract_text_content(message: Dict[str, Any]) -> str:
    """Extract all text content from a message.

    Handles both simple string content and multi-modal content arrays.

    Args:
        message: Message dictionary

    Returns:
        Concatenated text content
    """
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts)

    return str(content)


def has_images(message: Dict[str, Any]) -> bool:
    """Check if a message contains image content.

    Args:
        message: Message dictionary

    Returns:
        True if message contains images
    """
    content = message.get("content", "")

    if isinstance(content, list):
        return any(isinstance(part, dict) and part.get("type") in ("image_url", "image_file") for part in content)

    return False


def has_tool_calls(message: Dict[str, Any]) -> bool:
    """Check if a message contains tool calls.

    Args:
        message: Message dictionary

    Returns:
        True if message contains tool calls
    """
    return "tool_calls" in message and message["tool_calls"]


def count_tokens_estimate(messages: List[Dict[str, Any]]) -> int:
    """Estimate token count for a list of messages.

    This is a rough estimate based on text length. For accurate counts,
    use a tokenizer specific to your model.

    Args:
        messages: List of message dictionaries

    Returns:
        Estimated token count
    """
    total = 0

    for msg in messages:
        # Role and formatting overhead (~4 tokens per message)
        total += 4

        # Content tokens
        text = extract_text_content(msg)
        # Rough estimate: 1 token ≈ 4 characters
        total += len(text) // 4

        # Tool calls add tokens
        if has_tool_calls(msg):
            for tool_call in msg.get("tool_calls", []):
                func = tool_call.get("function", {})
                total += len(func.get("name", "")) // 4
                total += len(func.get("arguments", "")) // 4

    return total
