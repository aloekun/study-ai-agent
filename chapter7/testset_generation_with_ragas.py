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

testset.to_pandas()

# 出力例（マークダウンの表形式）
# 4 件の内、evolution_type で simple:2, reasoning:1, multi_context:1 になっている
# |index|question|contexts|ground\_truth|evolution\_type|metadata|episode\_done|
# |---|---|---|---|---|---|---|
# |0|What capabilities does NLP Cloud offer for users looking to utilize advanced AI engines?|\# NLPCloud

# \>\[NLP Cloud\]\(https://docs\.nlpcloud\.com/\#introduction\) is an artificial intelligence platform that allows you to use the most advanced AI engines, and even train your own engines with your own data\. 


# \#\# Installation and Setup

# - Install the `nlpcloud` package\.

# ```bash
# pip install nlpcloud
# ```

# - Get an NLPCloud api key and set it as an environment variable \(`NLPCLOUD\_API\_KEY`\)


# \#\# LLM

# See a \[usage example\]\(/docs/integrations/llms/nlpcloud\)\.

# ```python
# from langchain\_community\.llms import NLPCloud
# ```

# \#\# Text Embedding Models

# See a \[usage example\]\(/docs/integrations/text\_embedding/nlp\_cloud\)

# ```python
# from langchain\_community\.embeddings import NLPCloudEmbeddings
# ```
# |NLP Cloud offers users the capability to use the most advanced AI engines and even train their own engines with their own data\.|simple|\{'source': 'docs/docs/integrations/providers/nlpcloud\.mdx', 'file\_path': 'docs/docs/integrations/providers/nlpcloud\.mdx', 'file\_name': 'nlpcloud\.mdx', 'file\_type': '\.mdx', 'filename': 'docs/docs/integrations/providers/nlpcloud\.mdx'\}|true|
# |1|What type of software does Salesforce provide as a cloud-based solution?|\# Salesforce

# \[Salesforce\]\(https://www\.salesforce\.com/\) is a cloud-based software company that
# provides customer relationship management \(CRM\) solutions and a suite of enterprise
# applications focused on sales, customer service, marketing automation, and analytics\.

# \[langchain-salesforce\]\(https://pypi\.org/project/langchain-salesforce/\) implements
# tools enabling LLMs to interact with Salesforce data\.


# \#\# Installation and Setup

# ```bash
# pip install langchain-salesforce
# ```

# \#\# Tools

# See detail on available tools \[here\]\(/docs/integrations/tools/salesforce/\)\.
# |Salesforce provides customer relationship management \(CRM\) solutions and a suite of enterprise applications focused on sales, customer service, marketing automation, and analytics as a cloud-based solution\.|simple|\{'source': 'docs/docs/integrations/providers/salesforce\.mdx', 'file\_path': 'docs/docs/integrations/providers/salesforce\.mdx', 'file\_name': 'salesforce\.mdx', 'file\_type': '\.mdx', 'filename': 'docs/docs/integrations/providers/salesforce\.mdx'\}|true|
# |2|What needs to be set up for NLPCloud?|\# NLPCloud

# \>\[NLP Cloud\]\(https://docs\.nlpcloud\.com/\#introduction\) is an artificial intelligence platform that allows you to use the most advanced AI engines, and even train your own engines with your own data\. 


# \#\# Installation and Setup

# - Install the `nlpcloud` package\.

# ```bash
# pip install nlpcloud
# ```

# - Get an NLPCloud api key and set it as an environment variable \(`NLPCLOUD\_API\_KEY`\)


# \#\# LLM

# See a \[usage example\]\(/docs/integrations/llms/nlpcloud\)\.

# ```python
# from langchain\_community\.llms import NLPCloud
# ```

# \#\# Text Embedding Models

# See a \[usage example\]\(/docs/integrations/text\_embedding/nlp\_cloud\)

# ```python
# from langchain\_community\.embeddings import NLPCloudEmbeddings
# ```
# |To set up NLPCloud, you need to install the `nlpcloud` package and get an NLPCloud API key, which should be set as an environment variable \(`NLPCLOUD\_API\_KEY`\)\.|reasoning|\{'source': 'docs/docs/integrations/providers/nlpcloud\.mdx', 'file\_path': 'docs/docs/integrations/providers/nlpcloud\.mdx', 'file\_name': 'nlpcloud\.mdx', 'file\_type': '\.mdx', 'filename': 'docs/docs/integrations/providers/nlpcloud\.mdx'\}|true|
# |3|How does a vector DB improve semantic search and LLM retrieval, especially for doc embedding and quick queries?| cache with your LLMs:
# ```python
# from langchain\.globals import set\_llm\_cache
# import redis

# \# use any embedding provider\.\.\.
# from tests\.integration\_tests\.vectorstores\.fake\_embeddings import FakeEmbeddings

# redis\_url = "redis://localhost:6379"

# set\_llm\_cache\(RedisSemanticCache\(
#     embedding=FakeEmbeddings\(\),
#     redis\_url=redis\_url
# \)\)
# ```

# \#\# VectorStore

# The vectorstore wrapper turns Redis into a low-latency \[vector database\]\(https://redis\.com/solutions/use-cases/vector-database/\) for semantic search or LLM content retrieval\.

# ```python
# from langchain\_community\.vectorstores import Redis
# ```

# For a more detailed walkthrough of the Redis vectorstore wrapper, see \[this notebook\]\(/docs/integrations/vectorstores/redis\)\.

# \#\# Retriever

# The Redis vector store retriever wrapper generalizes the vectorstore class to perform 
# low-latency document retrieval\. To create the retriever, simply 
# call `\.as\_retriever\(\)` on the base vectorstore class\.

# \#\# Memory

# Redis can be used to persist LLM conversations\.

# \#\#\# Vector Store Retriever Memory

# For a more detailed walkthrough of the `VectorStoreRetrieverMemory` wrapper, see \[this notebook\]\(https://python\.langchain\.com/api\_reference/langchain/memory/langchain\.memory\.vectorstore\.VectorStoreRetrieverMemory\.html\)\.

# \#\#\# Chat Message History Memory
# For a detailed example of Redis to cache conversation message history, see \[this notebook\]\(/docs/integrations/memory/redis\_chat\_message\_history\)\.
# |The context does not provide specific details on how a vector database improves semantic search and LLM retrieval for document embedding and quick queries\.|multi\_context|\{'source': 'docs/docs/integrations/providers/redis\.mdx', 'file\_path': 'docs/docs/integrations/providers/redis\.mdx', 'file\_name': 'redis\.mdx', 'file\_type': '\.mdx', 'filename': 'docs/docs/integrations/providers/redis\.mdx'\}|true|
