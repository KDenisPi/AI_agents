"""
Minimal example: MCP client -> filesystem MCP server -> feed result to Ollama.

Setup:
    pip install mcp requests
    # The filesystem MCP server is a Node package, run via npx (needs Node.js installed):
    #   npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/dir
    # You don't need to run it manually - this script launches it for you via stdio.

What this does:
    1. Starts the official filesystem MCP server as a subprocess (stdio transport).
    2. Asks it what tools it exposes (list_directory, read_file, etc).
    3. Calls one tool (list_directory) to get real data.
    4. Stuffs that data into a prompt and sends it to your Ollama model
       (pointed at your Pi Zero W's Ollama instance, or localhost).
"""

import asyncio
import json
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Config ---
OLLAMA_URL = "http://<your-pi-ip>:11434/api/generate"  # change to your Pi's IP + port
OLLAMA_MODEL = "llama3.2:1b"                            # whatever model you have pulled
TARGET_DIR = "/home/claude/mcp_demo"                    # dir the MCP server is allowed to read


async def get_directory_listing_via_mcp(target_dir: str) -> str:
    """Connect to the filesystem MCP server and call its list_directory tool."""
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", target_dir],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover what tools this server offers
            tools = await session.list_tools()
            print("Available MCP tools:", [t.name for t in tools.tools])

            # Call the list_directory tool
            result = await session.call_tool(
                "list_directory", arguments={"path": target_dir}
            )
            # result.content is a list of content blocks (usually text)
            text_output = "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )
            return text_output


def ask_ollama(prompt: str) -> str:
    """Send a prompt to a local/remote Ollama instance."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"]


async def main():
    # Step 1: get real data from the MCP server
    listing = await get_directory_listing_via_mcp(TARGET_DIR)
    print("\n--- Raw MCP tool result ---")
    print(listing)

    # Step 2: hand that data to the model as context
    prompt = (
        f"Here is a directory listing:\n{listing}\n\n"
        "Summarize what kinds of files are in there in one sentence."
    )
    print("\n--- Asking Ollama ---")
    answer = ask_ollama(prompt)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())