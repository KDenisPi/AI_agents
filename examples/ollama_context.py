import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

# This list is your "session state" — you own it, Ollama doesn't store it
messages = [
    {"role": "system", "content": "You are a concise Linux/Raspberry Pi assistant."}
]

def ask(user_input):
    messages.append({"role": "user", "content": user_input})
    
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192}
    })
    
    reply = response.json()["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply

print(ask("How do I check disk usage on my Pi?"))
print(ask("What about just the SD card partition?"))
print(ask("And how do I set up a cron job to log that daily?"))
