from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain


@chain
def reverse_words(text: str) -> str:
    arr = text.split()
    arr.reverse()
    return " ".join(arr)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ]
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

chain = prompt | model | output_parser | reverse_words

output = chain.invoke({"input": "Hello World"})
print(output)

# 出力例
# today? you assist I can How Hello!
