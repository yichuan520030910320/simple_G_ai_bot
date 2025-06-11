"""
HuggingFace Chat Model Wrapper for vision models like Qwen2-VL
"""

import os
import base64
import requests
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field


class HuggingFaceChat(BaseChatModel):
    """Chat model wrapper for HuggingFace Inference API"""

    model: str = Field(description="HuggingFace model name")
    temperature: float = Field(default=0.0, description="Temperature for sampling")
    max_tokens: int = Field(default=1000, description="Max tokens to generate")
    api_token: Optional[str] = Field(default=None, description="HF API token")

    def __init__(self, model: str, temperature: float = 0.0, **kwargs):
        api_token = kwargs.get("api_token") or os.getenv("HUGGINGFACE_API_KEY")
        if not api_token:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is required")

        super().__init__(
            model=model, temperature=temperature, api_token=api_token, **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"

    def _format_message_for_hf(self, message: HumanMessage) -> Dict[str, Any]:
        """Convert LangChain message to HuggingFace format"""
        if isinstance(message.content, str):
            return {"role": "user", "content": message.content}

        # Handle multi-modal content (text + images)
        formatted_content = []
        for item in message.content:
            if item["type"] == "text":
                formatted_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                # Extract base64 data from data URL
                image_url = item["image_url"]["url"]
                if image_url.startswith("data:image"):
                    # Extract base64 data
                    base64_data = image_url.split(",")[1]
                    formatted_content.append({"type": "image", "image": base64_data})

        return {"role": "user", "content": formatted_content}

    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        """Generate response using HuggingFace Inference API"""

        # Format messages for HF API
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(self._format_message_for_hf(msg))

        # Prepare API request
        api_url = f"https://api-inference.huggingface.co/models/{self.model}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            return ChatResult(
                generations=[ChatGeneration(message=HumanMessage(content=content))]
            )

        except requests.exceptions.RequestException as e:
            # Fallback to simple text-only API if chat completions fail
            return self._fallback_generate(messages, **kwargs)

    def _fallback_generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        """Fallback to simple HF Inference API"""
        try:
            # Use simple inference API as fallback
            api_url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }

            # Extract text content only for fallback
            text_content = ""
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    if isinstance(msg.content, str):
                        text_content += msg.content
                    else:
                        for item in msg.content:
                            if item["type"] == "text":
                                text_content += item["text"] + "\n"

            payload = {
                "inputs": text_content,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                },
            }

            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "No response generated")
            else:
                content = "Error: Invalid response format"

            return ChatResult(
                generations=[ChatGeneration(message=HumanMessage(content=content))]
            )

        except Exception as e:
            # Last resort fallback
            error_msg = f"HuggingFace API Error: {str(e)}. Please check your API key and model availability."
            return ChatResult(
                generations=[ChatGeneration(message=HumanMessage(content=error_msg))]
            )
