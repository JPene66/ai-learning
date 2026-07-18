import os
from dotenv import load_dotenv
import json
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = 'gpt-4o-mini'

def ask_with_system(system_prompt, user_prompt, model=DEFAULT_MODEL, max_tokens=300, temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":user_prompt}
        ]
    )

    return response.choices[0].message.content

def ask_json(prompt, system_prompt = None, model=DEFAULT_MODEL, max_tokens=300):
    messages = []
    if system_prompt:
        messages.append({"role":"system", "content":system_prompt})
    messages.append({"role":"user", "content":prompt})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        response_format= {"type":"json_object"},
        messages = messages
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Warning: model did not return valid json", raw)
        return {}