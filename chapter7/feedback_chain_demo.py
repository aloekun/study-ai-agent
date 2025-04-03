from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from uuid import UUID
from langchain_core.tracers.context import collect_runs

import ipywidgets as widgets
from IPython.display import display
from langsmith import Client


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def display_feedback_buttons(run_id: UUID) -> None:
    # GoodボタンとBadボタンを準備
    good_button = widgets.Button(
        description="Good",
        button_style="success",
        icon="thumbs-up",
    )
    bad_button = widgets.Button(
        description="Bad",
        button_style="danger",
        icon="thumbs-down",
    )

    # クリックされた際に実行される関数を定義
    def on_button_clicked(button: widgets.Button) -> None:
        if button == good_button:
            score = 1
        elif button == bad_button:
            score = 0
        else:
            raise ValueError(f"Unknown button: {button}")

        client = Client()
        client.create_feedback(run_id=run_id, key="thumbs", score=score)
        print("フィードバックを送信しました")

    # ボタンがクリックされた際にon_button_clicked関数を実行
    good_button.on_click(on_button_clicked)
    bad_button.on_click(on_button_clicked)

    # ボタンを表示
    display(good_button, bad_button)


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()

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

chain = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
).assign(answer=prompt | model | StrOutputParser())

# LangSmithのトレースのID(Run ID)を取得するため、collect_runs関数を使用
with collect_runs() as runs_cb:
    output = chain.invoke("LangChainの概要を教えて")
    print(output["answer"])
    run_id = runs_cb.traced_runs[0].id

display_feedback_buttons(run_id)

# 出力例
# LangChainは、大規模言語モデル（LLM）を活用したアプリケーションを開発するためのフレームワークです。このフレームワークは、LLMアプリケーションのライフサイクルの各段階を簡素化します。具体的には、以下の3つの主要なステージがあります。
# 1. **開発**: LangChainのオープンソースコンポーネントやサードパーティの統合を使用してアプリケーションを構築します。LangGraphを利用することで、状態を持つエージェントを構築し、ストリーミングや人間の介入をサポートします。
# 2. **プロダクション化**: LangSmithを使用してアプリケーションを検査、監視、評価し、継続的に最適化して自信を持ってデプロイできるようにします。
# 3. **デプロイメント**: LangGraphアプリケーションをプロダクション対応のAPIやアシスタントに変換します。
# LangChainは、埋め込みモデルやベクターストアなどの関連技術に対する標準インターフェースを実装し、数百のプロバイダーと統合しています。また、複数のオープンソースライブラリで構成されており、さまざまな統合パッケージやコミュニティによって維持されるコンポーネントが含まれています。

# この部分に「Goodボタン」が表示される
# この部分に「Badボタン」が表示される
# 「Goodボタン」「Badボタン」を押すと、「フィードバックを送信しました」を表示する
