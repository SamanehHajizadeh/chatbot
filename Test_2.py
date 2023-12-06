from openai import OpenAI

client = OpenAI(api_key='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')


response = client.chat.completions.create(model="gpt-3.5-turbo",
messages=[{"role": "user", "content": "Say this is a test!"}],
temperature=0.7)

print(response)
