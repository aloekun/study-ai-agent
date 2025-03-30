from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


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

query_generation_prompt = ChatPromptTemplate.from_template("""\
質問に対してベクターデータベースからから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}
""")

# 質問に対して複数の検索クエリを生成する
# その後、retrieverを使用して、質問の回答に関連する文書を取得する
# この流れで検索すると、質問に関連するドキュメントを調べるだけより、期待する回答に近いドキュメントを拾いやすくなる
query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
)

# multi_query_retrieval_chainの結果に順位付けして回答する
rag_fusion_chain = {
    "question": RunnablePassthrough(),
    "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion,
} | prompt | model | StrOutputParser()

rag_fusion_chain.invoke("LangChainを使ってアプリ開発を促進するアイデアを提案して")

# 出力例
# LangChainを使ってアプリ開発を促進するアイデアとして、以下のようなプロジェクトを提案します。
# 1. **カスタマイズ可能なチャットボット**:
#    - LangChainの標準化されたチャットモデルインターフェースを利用して、特定の業界やニーズに応じたカスタマイズ可能なチャットボットを開発します。ユーザーは自分のビジネスに合わせたトピックや応答スタイルを選択できるようにします。
# 2. **情報検索エンジン**:
#    - LangChainのリトリーバー機能を活用して、特定のドキュメントやデータベースから情報を検索するアプリケーションを構築します。ユーザーが質問を入力すると、関連する情報を引き出して提供するシステムです。
# 3. **教育用アプリケーション**:
#    - LangChainを使って、学習者が質問をするとリアルタイムで回答を提供する教育用アプリを開発します。例えば、プログラミングや数学の問題に対する解説を行うことができます。
# 4. **多言語対応のカスタマーサポートシステム**:
#    - LangChainのチャットモデルを利用して、多言語に対応したカスタマーサポートシステムを構築します。ユーザーが自分の言語で質問をすると、適切な言語で回答を返すことができます。
# 5. **データ分析アシスタント**:
#    - LangChainを用いて、ユーザーが自然言語でデータ分析の質問をすると、SQLクエリを生成してデータベースから情報を引き出すアプリケーションを開発します。これにより、データ分析の専門知識がないユーザーでも簡単にデータを扱えるようになります。
# 6. **健康管理アプリ**:
#    - LangChainを活用して、ユーザーの健康に関する質問に答えるアプリを開発します。例えば、食事や運動に関するアドバイスを提供したり、症状に基づいて医療情報を提供することができます。
# これらのアイデアは、LangChainの機能を活かし、開発者が効率的にアプリケーションを構築できるようにすることを目的としています。
