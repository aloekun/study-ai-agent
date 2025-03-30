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

hypothetical_prompt = ChatPromptTemplate.from_template("""\
次の質問に回答する一文を書いてください。

質問: {question}
""")

# まず、hypothetical_promptを使用して、質問に対する仮の回答をドキュメントを見ずに生成する
# その後、ドキュメントから検索するretrieverを使用して、質問の回答に関連する文書を取得する
# この流れで検索すると、質問に関連するドキュメントを調べるだけより、期待する回答に近いドキュメントを拾いやすくなる
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

hyde_rag_chain = {
    "question": RunnablePassthrough(),
    "context": hypothetical_chain | retriever,
} | prompt | model | StrOutputParser()

hyde_rag_chain.invoke("LangChainでできることを教えて")

# 出力例

# LangChainでは、以下のようなことができます：
# 1. **標準化されたコンポーネントインターフェース**: 様々なAIアプリケーションのためのモデルや関連コンポーネントに対して、統一されたインターフェースを提供します。これにより、異なるプロバイダー間での切り替えが容易になります。
# 2. **オーケストレーション**: 複数のコンポーネントやモデルを組み合わせて、複雑なアプリケーションを構築するための効率的な接続をサポートします。これにより、複雑な制御フローや人間の介入が必要なアプリケーションを構築できます。
# 3. **可観測性と評価**: アプリケーションの動作を理解しやすくし、開発のペースを向上させるためのトレーシングや評価機能を提供します。これにより、開発者は自信を持ってアプリケーションを監視し、改善することができます。
# 4. **コンポーネントの選択と組み合わせ**: LangChainのエコシステム内のコンポーネントは独立して使用できるため、特定のニーズに応じて好きなコンポーネントを選んで組み合わせることができます。
# 5. **LangGraphの利用**: 複雑なアプリケーションのフローをノードとエッジのセットとして表現することで、高度な制御を可能にします。これにより、エージェントやマルチエージェントアプリケーションの構築が容易になります。
# 6. **LangSmithによる評価とトレーシング**: AIアプリケーションの観察と評価をサポートするプラットフォームを提供し、開発者がアプリケーションのパフォーマンスを迅速に評価できるようにします。
# これらの機能により、LangChainは開発者がAIアプリケーションを簡単に構築し、管理するための強力なツールとなっています。
