import gradio as gr
from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage

from core import __set_base_path__
from src.chat.chatbot import generate_message


def predict(message, history):
    chat_history = ChatMessageHistory()
    for human, ai in history:
        chat_history.add_user_message(HumanMessage(content=human))
        chat_history.add_ai_message(AIMessage(content=ai))

    message, token = generate_message(message, chat_history)
    print(token)

    return message

demo = gr.ChatInterface(
    fn=predict,
    autofocus=True,
)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
