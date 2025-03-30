from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever
from enum import Enum
from pydantic import BaseModel
from typing import Any
from langchain_core.documents import Document


class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def routed_retriever(inp: dict[str, Any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]

    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)
    
    raise ValueError(f"Unknown route: {route}")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()
# print(len(documents))


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)


prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()

# 読み込んだドキュメントから回答するretriever
langchain_document_retriever = retriever.with_config(
    {"run_name": "langchain_document_retriever"}
)

# webから回答するretriever
web_retriever = TavilySearchAPIRetriever(k=3).with_config(
    {"run_name": "web_retriever"}
)

# retrieverを振り分けるためのプロンプト
route_prompt = ChatPromptTemplate.from_template("""\
質問に回答するために適切なRetrieverを選択してください。

質問: {question}
""")

route_chain = (
    route_prompt
    | model.with_structured_output(RouteOutput)
    | (lambda x: x.route)
)

route_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "route": route_chain,
    }
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt | model | StrOutputParser()
)

route_rag_chain.invoke("LangChainの概要を教えて")

# 出力例(langchain_document_retrieverを選択した場合)
# LangChainは、大規模言語モデル（LLM）を活用したアプリケーションを開発するためのフレームワークです。このフレームワークは、LLMアプリケーションのライフサイクルの各ステージを簡素化します。具体的には、以下の3つの主要なステージがあります。
# 1. **開発**: LangChainのオープンソースコンポーネントやサードパーティの統合を使用してアプリケーションを構築します。LangGraphを利用することで、状態を持つエージェントを構築し、ストリーミングや人間の介入をサポートします。
# 2. **プロダクション化**: LangSmithを使用してアプリケーションを検査、監視、評価し、継続的に最適化して自信を持ってデプロイできるようにします。
# 3. **デプロイメント**: LangGraphアプリケーションをプロダクション対応のAPIやアシスタントに変換します。
# LangChainは、LLMや関連技術（埋め込みモデルやベクターストアなど）に対する標準インターフェースを実装し、数百のプロバイダーと統合しています。また、複数のオープンソースライブラリで構成されており、ユーザーは簡単にアプリケーションを構築、デプロイ、管理することができます。


# route_rag_chain.invoke("LangChainの概要を教えて")

# 出力例(web_retrieverを選択した場合)
# サンフランシスコの今日の天気は、晴れ時々雨または曇り時々雨の予報です。具体的な気温や湿度については、観測データが不定期に更新されるため、参考値としてご利用ください。
