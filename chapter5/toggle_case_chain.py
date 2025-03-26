from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def change_upper_lower(text: str) -> str:
    for chara in text:
        if chara.isupper():
            text = text.replace(chara, chara.lower())
        else:
            text = text.replace(chara, chara.upper())
    return text


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ]
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

chain = prompt | model | output_parser | change_upper_lower

output = chain.invoke({"input": "Hello World"})
print(output)

# 出力例
# hELLO! hOW CAN I ASSIST YOU TODAY?
