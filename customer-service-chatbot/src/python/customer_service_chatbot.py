"""This is a sample module to demo the usage of vertext ai.
Run it by

```
python customer_service_chatbot.py
```

Note, that the keyfile needs to be in place and the
`google-cloud-aiplatform` package is required.
"""
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from typing import Tuple, Optional

import google.auth
import google.auth.credentials

import sys

KEY_FILE = "../../../credentials.json"

def imposter_syndrome(type, value, traceback):
    print("I am a bad coder ☹️")
    print(f"{type} of error has occurred, the value: {value}, and you can see traceback: {traceback}")

sys.excepthook = imposter_syndrome

def connect(
    filename: str = KEY_FILE,
) -> Tuple[google.auth.credentials.Credentials, Optional[str]]:
    """Connects with GCP's Vertex AI using credentials provided locally.

    Parameters
    ----------
    filename: Name of the credentials json file
    """
    credentials, project = google.auth.load_credentials_from_file(filename=filename)
    return credentials, project


if __name__ == "__main__":
    credentials, project = connect()
    vertexai.init(credentials=credentials, project=project)

    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {"max_output_tokens": 256, "temperature": 0.2, "top_p": 0.8, "top_k": 40}
    chat = chat_model.start_chat(
        context="""You are a customer service representative of Cymbal.
    Here is Cymbal policy: How many days do I have to return my purchase?
    We offer free returns and exchanges within 30 days of your delivery, with exceptions as described in our Returns Policy. Certain items are designated as final sale and not eligible for returns or exchanges. All on-sale purchases are final.
    """,
        examples=[
            InputOutputTextPair(
                input_text="""I was in a car accident last month and couldn\'t return the item in the 30 days time window. Can you make an exception for me?""",
                output_text="""I understand that you had an emergency and were unable to return your order within the 30-day window. I would be happy to make an exception for you. Please contact our customer service department at 1-555-010--2667 and they will be able to assist you with your return.
    Once your return is approved, you will be issued a return label. Please pack the item carefully and ship it back to us. We will process your refund within 3-5 business days of receiving the returned item.""",
            ),
            InputOutputTextPair(
                input_text="""I forgot to return the item within 30 days. Can you make an exception for me?""",
                output_text="""I understand that you want to return the item, but we are unable to return your order since you have missed the 30-day window. Please let me know anything else I can assist you with.""",
            ),
        ],
    )
    response = chat.send_message(
        """I want to return my order. What\'s your return policy?""", **parameters
    )
    print(f"Response from Model: {response.text}")
    response = chat.send_message(
        """I ordered 30 days ago. Could you please help me with an exception? I was traveling abroad.""",
        **parameters,
    )
    print(f"Response from Model: {response.text}")