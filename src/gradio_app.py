import gradio as gr
from langchain.globals import set_verbose
from langchain.schema import AIMessage, HumanMessage

from lib import __set_base_path__
from src.chat.chatbot import generate_message

set_verbose(True)

template = """
    あなたは高性能チャットbotです。
    ユーザーからの質問に誠実に回答してください。
    また、あなたの回答によって不利益が生じる者が出ないよう注意を払ってください。

    ユーザーの入力: {input}
    """


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    partial_message = ""

    for chunk in generate_message(message):
        partial_message += chunk
        yield partial_message


gr.ChatInterface(predict).launch()
