from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 読み込むファイルの種類をフィルタリングする関数
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".md")


# Gitリポジトリからドキュメントを読み込む
# clone_url: リポジトリのURL（例として自分のリポジトリを使用）
# repo_path: リポジトリをクローンするパス（ローカルの保存パス、任意の名前をつけてよい）
# branch: リポジトリのブランチ
# file_filter: ファイルをフィルタリングする関数
loader = GitLoader(
    clone_url="https://github.com/aloekun/docker-kubernetes-practice",
    repo_path="./docker_note",
    branch="master",
    file_filter=file_filter,
)

raw_docs = loader.load()
# print(len(raw_docs))

# テキストをチャンク（扱いやすい単位）に分割する
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(raw_docs)

# OpenAIのテキスト埋め込みを使用して、ベクター化する
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query = "Kubernetesの使い方の記載はありますか？"

# ドキュメントと埋め込んだベクターからVector storeを初期化
db = Chroma.from_documents(docs, embeddings)
# Vector storeから検索するRetrieverを作成
retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈："""
{context}
"""

質問：{question}
''')

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Chainを使って検索結果を出力する
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

output = chain.invoke(query)
print(output)

# 出力例
# はい、Kubernetesの使い方についての記載があります。具体的には、ymlファイルを使用してデプロイを実行する方法や、ポッドの状態を確認するためのコマンド（`kubectl get pods`）が紹介されています。また、ymlファイルを適用するためのコマンド（`kubectl apply -f [ファイルパス]`）も示されています。
