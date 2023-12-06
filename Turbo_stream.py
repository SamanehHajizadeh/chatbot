from openai import OpenAI

OPENAI_API_KEY= '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'

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content)