from google import genai
from google.genai.types import GenerateContentConfig
import os
from dotenv import load_dotenv

load_dotenv()

llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

tools = []
user_message = {"role": "user", "content": "Wer bist du?"}
messages = [user_message]



test = llm.models.generate_content(
    model="gemini-2.5-flash",
    contents=messages,
    config=GenerateContentConfig(
        tools=tools,
        max_output_tokens=1000
    ),
)
print(test)
