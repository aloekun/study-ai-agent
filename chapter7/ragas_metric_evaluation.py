from typing import Any

from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM
from ragas.metrics import answer_relevancy, context_precision


class RagasMetricEvaluator:
    def __init__(self, metric: Metric, llm: BaseChatModel, embeddings: Embeddings):
        self.metric = metric

        # LLMとEmbeddingsをMetricに設定
        if isinstance(self.metric, MetricWithLLM):
            self.metric.llm = LangchainLLMWrapper(llm)
        if isinstance(self.metric, MetricWithEmbeddings):
            self.metric.embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        context_strs = [doc.page_content for doc in run.outputs["contexts"]]

        # Ragasの評価メトリクスのScoreメソッドでスコアを算出
        # 質問（question）・実際の回答（answer）・実際の検索結果（contexts）・期待する回答（ground_truth）
        score = self.metric.score(
            {
                "question": example.inputs["question"],
                "answer": run.outputs["answer"],
                "contexts": context_strs,
                "ground_truth": example.outputs["ground_truth"],
            },
        )
        return {"key": self.metric.name, "score": score}


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


# Chainを実行して、評価に必要な情報を辞書形式に変換して返す
def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    question = inputs["question"]
    output = chain.invoke(question)
    return {
        "contexts": output["context"],
        "answer": output["answer"],
    }


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()

# 評価に使うメトリクス
metrics = [context_precision, answer_relevancy]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

evaluators = [
    RagasMetricEvaluator(metric, llm, embeddings).evaluate
    for metric in metrics
]

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

# 評価を実行、LangSmithに評価結果を保存
evaluate(
    predict,
    data="agent-book",
    evaluators=evaluators,
)

# 出力なし
# LangSmith に Ragas の評価メトリクスのグラフが表示される
