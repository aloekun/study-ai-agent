from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


retriever = TavilySearchAPIRetriever(k=3)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

output = chain.invoke("ニューヨークの今日の天気は？")
print(output)

# 出力例
# 文脈には具体的な今日の天気の情報は含まれていませんが、ニューヨークの今日の天気については、日本気象協会やAccuWeather、ウェザーニュースのサイトで確認できると記載されています。具体的な天気の詳細はそれらのサイトを参照してください。
