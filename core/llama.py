import requests
import json


class Llama:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/chat"

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message if isinstance(message, str) else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        assistant_message = {
            "role": "assistant",
            "content": message if isinstance(message, str) else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: dict):
        if isinstance(message, dict) and "message" in message:
            return message["message"].get("content", "")
        return ""

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> dict:
        # Note: Llama via Ollama doesn't support tools in the same way as Claude
        # This is a basic implementation that mirrors Claude's interface
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }

        if system:
            # Insert system message at the beginning if not already present
            if not messages or messages[0].get("role") != "system":
                payload["messages"] = [{"role": "system", "content": system}] + messages

        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # Create a response object that mimics Claude's Message structure
            # For Llama/Ollama, we wrap the response
            return {
                "message": result.get("message", {}),
                "stop_reason": "end_turn" if result.get("done") else "stop",
                "content": [{"type": "text", "text": result.get("message", {}).get("content", "")}],
            }
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Llama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            raise RuntimeError(f"Error calling Llama API: {e}")
