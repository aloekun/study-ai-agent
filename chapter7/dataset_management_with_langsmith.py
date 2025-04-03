from langsmith import Client
from langchain_community.document_loaders import GitLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import nest_asyncio
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context


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

for document in documents:
    document.metadata["filename"] = document.metadata["source"]

nest_asyncio.apply()

# Ragasを使って、テストデータのgeneratorを作成
generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o-mini"),
    critic_llm=ChatOpenAI(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings(),
)

# テストデータのgeneratorで、テストの数を4件作る
# テストを作る都度 LLM を呼び出すので、モデルに使うエージェントや呼び出し回数に注意、呼び出し料金がかさむ
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=4,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)

dataset_name = "agent-book"

client = Client()

if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)

dataset = client.create_dataset(dataset_name=dataset_name)

inputs = []
outputs = []
metadatas = []

for testset_record in testset.test_data:
    inputs.append(
        {
            "question": testset_record.question,
        }
    )
    outputs.append(
        {
            "contexts": testset_record.contexts,
            "ground_truth": testset_record.ground_truth,
        }
    )
    metadatas.append(
        {
            "source": testset_record.metadata[0]["source"],
            "evolution_type": testset_record.evolution_type,
        }
    )

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    metadata=metadatas,
    dataset_id=dataset.id,
)

# 出力なし
# LangSmith に Ragas から作ったテストデータを登録する
