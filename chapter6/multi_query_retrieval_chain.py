from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


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


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


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

multi_query_rag_chain = {
    "question": RunnablePassthrough(),
    "context": query_generation_chain | retriever.map(),
} | prompt | model | StrOutputParser()

multi_query_rag_chain.invoke("LangChainを使って作れるアプリのアイデアを提案して")

# 出力例
# LangChainを使って作れるアプリのアイデアをいくつか提案します。
# 1. **カスタマイズ可能なチャットボット**:
#    - ユーザーが特定のニーズに合わせてカスタマイズできるチャットボットを作成します。例えば、特定の業界（医療、教育、カスタマーサポートなど）に特化した知識を持つボットを構築し、ユーザーが質問をすると適切な情報を提供します。
# 2. **文書検索エンジン**:
#    - PDFやWord文書などの非構造化データを対象にした文書検索エンジンを開発します。ユーザーが質問を入力すると、関連する文書を検索し、要約や重要な情報を抽出して表示します。
# 3. **自動要約ツール**:
#    - 長文のテキストを自動的に要約するアプリを作成します。ニュース記事や研究論文などを入力すると、重要なポイントを短くまとめて提供します。
# 4. **質問応答システム**:
#    - 特定のデータベースや知識ベースに基づいて、ユーザーの質問に対して正確な回答を提供するシステムを構築します。例えば、SQLデータベースを利用して、ユーザーがデータに基づいた質問をすると、適切なSQLクエリを実行して結果を返します。
# 5. **学習支援アプリ**:
#    - 学生向けに、特定の科目に関する質問に答えたり、学習資料を提供したりするアプリを開発します。ユーザーが質問をすると、関連する教材やリソースを提案します。
# 6. **感情分析ツール**:
#    - ソーシャルメディアの投稿やレビューを分析し、感情を評価するアプリを作成します。ユーザーが特定のトピックについての投稿を入力すると、その感情をポジティブ、ネガティブ、ニュートラルに分類します。
# 7. **パーソナライズされたニュースフィード**:
#    - ユーザーの興味に基づいてニュースを収集し、パーソナライズされたニュースフィードを提供するアプリを開発します。ユーザーが興味のあるトピックを選択すると、それに関連する最新のニュースを表示します。
# これらのアイデアは、LangChainの機能を活用して、さまざまなニーズに応じたアプリケーションを開発するための出発点となります。
