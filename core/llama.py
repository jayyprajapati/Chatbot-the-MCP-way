import requests
import json
import uuid


class Llama:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/chat"

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message if isinstance(message, str) else json.dumps(message) if isinstance(message, list) else message,
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

    def _convert_tools_to_ollama_format(self, tools: list) -> list:
        """Convert tools from Claude/MCP format to Ollama format."""
        if not tools:
            return []
        
        ollama_tools = []
        for tool in tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            }
            ollama_tools.append(ollama_tool)
        return ollama_tools

    def _convert_ollama_tool_calls_to_claude_format(self, tool_calls: list) -> list:
        """Convert Ollama tool calls to Claude-compatible format."""
        content = []
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            # Parse arguments - Ollama returns them as a dict or string
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            
            content.append({
                "type": "tool_use",
                "id": f"tool_{uuid.uuid4().hex[:8]}",
                "name": func.get("name", ""),
                "input": args
            })
        return content

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

        # Add tools to payload if provided
        if tools:
            payload["tools"] = self._convert_tools_to_ollama_format(tools)

        try:
            print(f"[DEBUG] Sending request to Ollama with {len(tools) if tools else 0} tools")
            response = requests.post(self.api_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            message = result.get("message", {})
            tool_calls = message.get("tool_calls", [])
            text_content = message.get("content", "")
            
            print(f"[DEBUG] Ollama response - has tool_calls: {len(tool_calls) > 0}, content length: {len(text_content)}")
            
            # Check if Ollama wants to use tools
            if tool_calls:
                print(f"[DEBUG] Tool calls detected: {[tc.get('function', {}).get('name') for tc in tool_calls]}")
                # Convert tool calls to Claude-compatible format
                content = self._convert_ollama_tool_calls_to_claude_format(tool_calls)
                # Add any text content as well
                if text_content:
                    content.insert(0, {"type": "text", "text": text_content})
                
                return {
                    "message": message,
                    "stop_reason": "tool_use",
                    "content": content,
                }
            else:
                # No tool calls, just text response
                return {
                    "message": message,
                    "stop_reason": "end_turn" if result.get("done") else "stop",
                    "content": [{"type": "text", "text": text_content}],
                }
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Llama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            raise RuntimeError(f"Error calling Llama API: {e}")
