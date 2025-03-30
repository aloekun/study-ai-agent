from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


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

chain = {
    "question": RunnablePassthrough(),
    "context": retriever,
} | prompt | model | StrOutputParser()

chain.invoke("LangChainの強み・弱みを教えて")

# 出力例
# LangChainの強みと弱みは以下の通りです。
# ### 強み
# 1. **標準化されたコンポーネントインターフェース**: LangChainは、さまざまなAIアプリケーションに必要なコンポーネントの標準インターフェースを提供しており、異なるプロバイダー間での切り替えが容易です。
# 2. **オーケストレーション機能**: 複数のコンポーネントやモデルを効率的に接続し、複雑なアプリケーションフローを構築するためのオーケストレーション機能を提供しています。特に、LangGraphを使用することで、状態を持つマルチアクターアプリケーションを構築できます。
# 3. **観測性と評価**: LangSmithを通じて、アプリケーションのトレースや評価が可能で、開発者がアプリケーションの動作を理解しやすくなります。
# 4. **非同期および並列実行のサポート**: LCELを使用することで、非同期および並列実行が可能になり、パフォーマンスの向上が期待できます。
# 5. **簡素化されたストリーミング**: LCELチェーンはストリーミングをサポートしており、出力を逐次的に生成することができます。

# ### 弱み
# 1. **複雑な状態管理や分岐が必要な場合の制約**: LCELはシンプルなオーケストレーションタスクに適しているが、複雑な状態管理や分岐、サイクルが必要な場合にはLangGraphを使用することが推奨されており、LCELの使用が制限されることがあります。
# 2. **学習曲線**: LangChainの多機能性は、特に初心者にとっては学習曲線が急である可能性があります。さまざまなコンポーネントやインターフェースを理解する必要があります。
# 3. **レガシーチェーンからの移行の難しさ**: 既存のレガシーチェーンからLCELへの移行には、ガイドラインに従う必要があり、手間がかかる場合があります。
# 4. **特定のユースケースに対する最適化の不足**: 一部の特定のユースケースに対しては、LangChainが最適化されていない場合があり、他のフレームワークやライブラリの方が適していることがあります。
# これらの強みと弱みを考慮しながら、LangChainを使用するかどうかを判断することが重要です。
