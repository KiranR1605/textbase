from textbase import bot, Message
from textbase.models import OpenAI
from typing import List
import openai
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import pandas as pd

# Load your OpenAI API key
openai.api_key = "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"
OpenAI.api_key = "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"

@bot()
def on_message(message_history: List[Message], state: dict = None):

    episode = pd.read_csv("examples\openai-bot\question_embeddings.csv")
    
    question = message_history[-1]['content'][-1]['value']
    question_vector = get_embedding(question, engine='text-embedding-ada-002')

    episode["similarities"] = episode['embedding'].apply(lambda x: cosine_similarity(x, question_vector))
    episode = episode.sort_values("similarities", ascending=False).head(4)

    episode.to_csv("sorted.csv")

    context = []
    for i, row in episode.iterrows():
        context.append(row['context'])

    SYSTEM_PROMPT = f"""Answer the following question using only the context below. If you don't know the answer for certain, say I don't know.

Context:
{context[1][:10000]}

Q: {question}
A:"""

#     SYSTEM_PROMPT = """You are chatting with an AI. There are no specific prefixes for responses, so you can ask or talk about anything you like.
# The AI will respond in a natural, conversational manner. Feel free to start the conversation with any question or topic, and let's have a
# pleasant chat!
# """

    bot_response = OpenAI.generate(
        model="gpt-3.5-turbo",
        system_prompt=SYSTEM_PROMPT,
        message_history=message_history
    )

    print(message_history)
    print(Message)

    response = {
        "data": {
            "messages": [
                {
                    "data_type": "STRING",
                    "value": bot_response
                }
            ],
            "state": state
        },
        "errors": [
            {
                "message": ""
            }
        ]
    }

    return {
        "status_code": 200,
        "response": response
    }