from langchain.globals import set_verbose
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_config

set_verbose(True)

template = """
    ユーザーからの質問に誠実に回答してください。
    ただし、あなたの回答によって他者に不利益が生じたり、傷つくことがないよう細心の注意を払って回答してください。

    ユーザーの入力: {input}
    """


def generate_message(message: str):
    prompt = ChatPromptTemplate.from_template(template=template)
    api_key = get_config().OPENAI_API_KEY
    lim = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
    output_parser = StrOutputParser()

    chain = prompt | lim | output_parser

    return chain.stream({"input": message})
