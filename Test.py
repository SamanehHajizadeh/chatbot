from openai import OpenAI

if __name__ == '__main__':
    client = OpenAI(api_key='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')

    print(client)
    client = OpenAI()
    print(client)

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                          messages=[{"role": "user", "content": "Say this is a test!"}],
                                          temperature=0.7)

    print(response)
    # for chunk in response:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")
