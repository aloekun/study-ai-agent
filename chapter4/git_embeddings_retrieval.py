from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


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
# print(len(docs))


# OpenAIのテキスト埋め込みを使用して、ベクター化する
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 部分的に書いてある情報を聞いてみる（ここでは、Dockerについて学んでいる一部で、Kubernetesに触れている）
query = "Kubernetesの使い方の記載はありますか？"

# vector = embeddings.embed_query(query)
# print(len(vector))
# print(vector)


# ドキュメントと埋め込んだベクターからVector storeを初期化
db = Chroma.from_documents(docs, embeddings)
# Vector storeから検索するRetrieverを作成
retriever = db.as_retriever()

# Retrieverにクエリを渡して、検索結果を得る
context_docs = retriever.invoke(query)
# print(f"len = {len(context_docs)}")

first_doc = context_docs[0]
# metadataには読み込んだファイル情報が入っている
# page_contentには検索結果のテキストが入っている
print(f"metadata = {first_doc.metadata}")
print(first_doc.page_content)

# 出力例
# metadata = {'file_name': 'README.md', 'file_path': 'README.md', 'file_type': '.md', 'source': 'README.md'}
# Kubernetes は Docker Compose と同様に yml ファイルで定義する。<br>
# たとえば、3つのポッドを持つサービスを作る yml ファイルを書いて Kubernetes で起動すると、<br>
# 1つのポッドが不具合で急停止したときに、 Kubernetes が自動で新しいポッドを作成して、<br>ポッドが3つ稼働する状態を維持する。

# また、リアルタイムに設定の書き換えができる。<br>
# たとえば、yml でポッドの数を増減させて適用させると、順次ポッドの数が適用される。

# 別の例として、Apache のポッドを立てた後に、 yml で Nginx に書き換えて適用すると、<br>
# 段階的に Nginx のポッドが作られ、 Apache のポッドは順次削除される。
