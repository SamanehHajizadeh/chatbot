import os

from openai import OpenAI

client = OpenAI(api_key='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')

OpenAI_Organization = "org-H1TADqTdsj4WIS2EvKElAD1h"
Open_AI_Key = "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"

os.environ['OPENAI_API_KEY'] = 'your-api-key'
Model_ = "gpt-3.5-turbo"
Model = "gpt-3.5-turbo-1106"

if __name__ == '__main__':

    response = client.chat.completions.create(model=Model,
                                              messages=[{"role": "user", "content": "Say this is a test!"}],
                                              temperature=0.7)

    print(response)
    for chunk in response:
        print(chunk)
