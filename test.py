import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("groq_QA_key")  # store your key in .env

client = Groq(api_key=api_key)
print(client.models.list())
