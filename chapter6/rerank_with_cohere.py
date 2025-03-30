from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from typing import Any
from langchain_cohere import CohereRerank
from langchain_core.documents import Document


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
    return cohere_reranker.compress_documents(documents=documents, query=question)


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

# Cohereを使って、retrieverから取得したドキュメントを再ランク付けする
rerank_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "documents": retriever,
    }
    | RunnablePassthrough.assign(context=rerank)
    | prompt | model | StrOutputParser()
)

rerank_rag_chain.invoke("LangChainに最近追加された機能を教えて")

# 出力例
# LangChainに最近追加された機能についての具体的な情報は文脈には含まれていませんが、LangChainは新しいコンポーネントや機能を継続的に追加していることが示唆されています。特に、LangChain Expression Language (LCEL)やLangGraphのような新しいオーケストレーションソリューションが導入されており、これによりアプリケーションの開発や管理がより効率的に行えるようになっています。
# 具体的な新機能については、公式のドキュメントやリリースノートを参照することをお勧めします。
