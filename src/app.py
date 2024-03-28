import gradio as gr
from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from anyio.from_thread import start_blocking_portal

from core import __set_base_path__
from src.chat.chatbot import generate_message, StreamingCallbackHandler


def predict(message, history):
    callback_handler = StreamingCallbackHandler()
    chat_history = ChatMessageHistory()
    for human, ai in history:
        chat_history.add_user_message(HumanMessage(content=human))
        chat_history.add_ai_message(AIMessage(content=ai))
        
    with start_blocking_portal() as portal:
        portal.start_task_soon(generate_message, message, chat_history, callback_handler)
        
        content = ""
        while True:
            next_token = callback_handler.que.get()
            if next_token is None:
                break
            content += next_token

            yield content
        


demo = gr.ChatInterface(
    fn=predict,
    autofocus=True,
)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
