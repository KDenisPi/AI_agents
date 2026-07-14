from ollama import Client

# Initialize the client with your remote server's IP and port
client = Client(host='http://192.168.1.57:11434')

# Call the chat function using the client instance
response = client.chat(
    model='qwen3.6:latest',
    messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ],
)

print(response.message.content)
