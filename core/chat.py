from core.llama import Llama
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    def __init__(self, llama_service: Llama, clients: dict[str, MCPClient]):
        self.llama_service: Llama = llama_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[dict] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            response = self.llama_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )

            # For tool_use, we need to add the assistant message with tool_calls info
            if response.get("stop_reason") == "tool_use":
                # Add assistant message with the original message (contains tool_calls)
                assistant_msg = response.get("message", {})
                self.messages.append({"role": "assistant", "content": assistant_msg.get("content", ""), "tool_calls": assistant_msg.get("tool_calls", [])})
                
                print(f"[DEBUG] LLM wants to use tools")
                if response.get("message", {}).get("content"):
                    print(f"[DEBUG] LLM says: {response.get('message', {}).get('content')}")
                
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )
                
                print(f"[DEBUG] Tool results: {tool_result_parts}")

                # Add tool results as tool messages (Ollama format)
                for result in tool_result_parts:
                    tool_msg = {
                        "role": "tool",
                        "content": result.get("content", ""),
                    }
                    self.messages.append(tool_msg)
            else:
                self.llama_service.add_assistant_message(self.messages, response.get("message", {}).get("content", ""))
                final_text_response = self.llama_service.text_from_message(
                    response
                )
                break

        return final_text_response
