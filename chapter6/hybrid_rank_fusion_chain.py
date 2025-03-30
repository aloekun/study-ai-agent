from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableParallel
from langchain_core.documents import Document


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]],
    k: int = 60,
) -> list[str]:
    # 各ドキュメントのコンテンツ（文字列）とそのスコアの対応を保持する辞書を準備
    content_score_mapping = {}

    # 検索クエリごとにループ
    for docs in retriever_outputs:
        # 検索結果のドキュメントごとにループ
        for rank, doc in enumerate(docs):
            content = doc.page_content

            # 初めて登場したコンテンツの場合はスコアを0で初期化
            if content not in content_score_mapping:
                content_score_mapping[content] = 0
            
            # (1 / (順位 + k)) のスコアを加筆
            content_score_mapping[content] += 1 / (rank + k)
    
    # スコアの大きい順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]


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

# ドキュメントから回答するretriever
chroma_retriever = retriever.with_config(
    {"run_name": "chroma_retriever"}
)

# BM25Retrieverを使用してドキュメントから回答するretriever
bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    {"run_name": "bm25_retriever"}
)

# 2つのretrieverの回答を順位付けして回答するretriever
hybrid_retriever = (
    RunnableParallel(
        {
            "chroma_documents": chroma_retriever,
            "bm25_documents": bm25_retriever,
        }
    )
    | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
    | reciprocal_rank_fusion
)

hybrid_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hybrid_retriever,
    }
    | prompt | model | StrOutputParser()
)

hybrid_rag_chain.invoke("LangChainの得意なこと・苦手なことを教えて")

# 出力例
# LangChainの得意なことと苦手なことは以下の通りです。

# ### 得意なこと
# 1. **シンプルなオーケストレーション**: LangChainは、LCELを使用してシンプルなチェーンを構築するのに適しています。例えば、単一のLLM呼び出しや、簡単なプロンプトとLLMの組み合わせなどです。
# 2. **最適化された実行**: LCELを使用することで、チェーンの実行が最適化され、並列実行や非同期処理が可能になります。これにより、処理のレイテンシが大幅に削減されます。
# 3. **ストリーミング出力**: LCELチェーンはストリーミングが可能で、実行中にインクリメンタルな出力を提供します。
# 4. **トレーシングとデバッグ**: LangSmithとの統合により、チェーンの各ステップが自動的にログに記録され、可視性とデバッグの容易さが向上します。
# 5. **標準APIの提供**: すべてのチェーンがRunnableインターフェースを使用して構築されるため、他のRunnableと同様に使用できます。

# ### 苦手なこと
# 1. **複雑なオーケストレーション**: 複雑な状態管理、分岐、サイクル、複数のエージェントを必要とするアプリケーションには不向きです。このような場合は、LangGraphを使用することが推奨されます。
# 2. **大規模なチェーン**: 数百ステップのチェーンを運用することは可能ですが、一般的にはシンプルなオーケストレーションタスクに使用することが推奨されています。
# 3. **直接的なLLM呼び出し**: 単一のLLM呼び出しを行う場合、LCELを使用する必要はなく、直接チャットモデルを呼び出す方が適切です。

# これらの特性を考慮して、LangChainを使用する際には、アプリケーションの要件に応じた適切な選択を行うことが重要です。
