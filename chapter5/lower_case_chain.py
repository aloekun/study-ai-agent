from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda


# テキストを小文字に変換する関数
def lower(text: str) -> str:
    return text.lower()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ]
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

chain = prompt | model | output_parser | RunnableLambda(lower)

output = chain.invoke({"input": "Hello World"})
print(output)

# 出力例
# hello! how can i assist you today?
