from typing import Tuple, Optional

import google.auth
import google.auth.credentials
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair

KEY_FILE = "../credentials.json"

context = """As a travel agency we offers some ancillary products to customer.
Ancillary products are:
1. Travel Insurance 
2. baggage
3. Seatmap
4. simple Visa
5. Rentable car
6. cabin baggage
Answer with one Ancillary product.
Serve as a travel agency offering various ancillary products to customers.  
If there are more than three ancillary products available, limit your questions to three.
"""

# Move the examples outside and use them dynamically
examples = [
    InputOutputTextPair(
        input_text="""Are you interested in protecting your trip in case of unexpected events?""",
        output_text="""  Travel Insurance """,
    ),
    InputOutputTextPair(
        input_text="""Do you need to check in any baggage during your trip? If so, would you like to purchase baggage allowance??""",
        output_text="""baggage, cabin baggage""",
    ),
    InputOutputTextPair(
        input_text="""Do you know for entering to USA you have to have visa??""",
        output_text="""simple Visa""",
    ),
]


def connect(filename: str = KEY_FILE) -> Tuple[google.auth.credentials.Credentials, Optional[str]]:
    """Connects with GCP's Vertex AI using credentials provided locally."""
    credentials, project = google.auth.load_credentials_from_file(filename=filename)
    return credentials, project


def get_user_input(prompt: str) -> str:
    """Get user input with a prompt."""
    return input(prompt).strip()


def interact_with_model_(chat: ChatModel, parameters: dict):
    """Interact with the chat model."""
    while True:
        user_input = get_user_input(chat)
        if user_input.lower() == 'exit':
            break

        response = chat.send_message(user_input, **parameters)
        print(f"Response from Model: {response.text}")


def interact_with_model(chat: ChatModel, parameters: dict):
    """Interact with the chat model."""
    user_input = get_user_input("Ask a question or type 'exit' to end: ")
    if user_input.lower() == 'exit':
        return

    response = chat.send_message("""There can be unexpected missing flight during trip?""", **parameters)
    print(f"Response from Model: {response.text}")

    response = chat.send_message("""2 kid are following and lots of stuff is with customer.""", **parameters)
    print(f"Response from Model: {response.text}")


    response = chat.send_message("""A European citizen.""",**parameters)
    print(f"Response from Model: {response.text}")

    response = chat.send_message("""There can be needs to going to a lot of places with long distance between?""", **parameters)
    print(f"Response from Model: {response.text}")

    response = chat.send_message("""There can be more than one person in trip and they need to sit beside each other.""", **parameters)
    print(f"Response from Model: {response.text}")

    response = chat.send_message("""There can be a long trip with a lot of stuff to checkin them.""",**parameters)
    print(f"Response from Model: {response.text}")


if __name__ == "__main__":
    credentials, project = connect()
    vertexai.init(credentials=credentials, project=project)

    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {"max_output_tokens": 256, "temperature": 0.2, "top_p": 0.8, "top_k": 40}

    chat = chat_model.start_chat(context=context, examples=examples)
    interact_with_model(chat, parameters)
