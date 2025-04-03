import operator
from typing import Annotated
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class State(BaseModel):
    query: str = Field(
        ..., description="ユーザーからの質問"
    )
    current_role: str = Field(
        default="", description="選定された回答ロール"
    )
    messages: Annotated[list[str], operator.add] = Field(
        default=[], description="回答履歴"
    )
    current_judge: bool = Field(
        default=False, description="品質チェックの結果"
    )
    judgement_reason: str = Field(
        default="", description="品質チェックの判定理由"
    )


class Judgement(BaseModel):
    reason: str = Field(default="", description="判定理由")
    judge: bool = Field(default=False, description="判定結果")


def selection_node(state: State) -> dict[str, Any]:
    query = state.query
    role_options = "\n".join([f"{k}. {v['name']}: {v['description']}" for k, v in ROLES.items()])
    prompt = ChatPromptTemplate.from_template("""
質問を分析し、最も適切な回答担当ロールを選択してください。

選択肢:
{role_options}

回答は選択肢の番号（1、2、または3）のみを返してください。

質問: {query}
""".strip()
    )
    # 選択肢の番号のみを返すことを期待したいため、max_tokens の値を 1 に変更
    chain = prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    role_number = chain.invoke({"role_options": role_options, "query": query})

    selected_role = ROLES[role_number.strip()]["name"]
    return {"current_role": selected_role}


def answering_node(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
    prompt = ChatPromptTemplate.from_template("""
あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。

役割の詳細:
{role_details}

質問: {query}

回答:""".strip()
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"role": role, "role_details": role_details, "query": query})
    return {"messages": [answer]}


def check_node(state: State) -> dict[str, Any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template("""
以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。また、その判定理由も説明してください。

ユーザーからの質問: {query}
回答: {answer}
""".strip()
    )
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke({"query": query, "answer": answer})

    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason
    }


ROLES = {
    "1": {
        "name": "一般知識エキスパート",
        "description": "幅広い分野の一般的な質問に答える",
        "details": "幅広い分野の一般的な質問に対して、正確でわかり易い回答を提供してください。"
    },
    "2": {
        "name": "生成AI製品エキスパート",
        "description": "生成AIや関連製品、技術に関する専門的な質問に答える",
        "details": "生成AIや関連製品、技術に関する専門的な質問に対して、最新の情報と深い洞察を提供してください。"
    },
    "3": {
        "name": "カウンセラー",
        "description": "個人的な悩みや心理的な問題に対してサポートを提供する",
        "details": "個人的な悩みや心理的な問題に対して、共感的で支援的な回答を提供し、可能であれば適切なアドバイスも行ってください。"
    }
}

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# 後からmax_tokensの値を変更できるように、変更可能なフィールドを宣言
llm = llm.configurable_fields(max_tokens=ConfigurableField(id='max_tokens'))

workflow = StateGraph(State)
workflow.add_node("selection", selection_node)
workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

# selection ノードから処理を開始
workflow.set_entry_point("selection")

# selection ノードから answering ノードへ
workflow.add_edge("selection", "answering")
# answering ノードから check ノードへ
workflow.add_edge("answering", "check")

# check ノードから次のノードへの遷移に条件付きエッジを定義
# state.current_judge の値が True なら END ノードへ、False なら selection ノードへ
workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,
    {True: END, False: "selection"}
)

compiled = workflow.compile()

initial_state = State(query="生成AIについて教えてください")
result = compiled.invoke(initial_state)

# print(result)
# 出力例
# 生成AI製品エキスパートとしてお答えします。
# 生成AIとは、人工知能の一分野であり、テキスト、画像、音声、動画などのコンテンツを生成する能力を持つモデルを指します。これらのモデルは、大量のデータを学習し、そのパターンを理解することで、新しいコンテンツを生成することができます。
# 代表的な生成AIの技術には、以下のようなものがあります：
# 1. **自然言語処理（NLP）モデル**: 例えば、GPT（Generative Pre-trained Transformer）シリーズは、テキストの生成や翻訳、要約などに利用されます。これらのモデルは、人間のように自然な文章を生成することが可能です。
# 2. **画像生成モデル**: GAN（Generative Adversarial Networks）やVAE（Variational Autoencoders）などの技術を用いて、新しい画像を生成します。これにより、アート作品の創作や、写真の修正、合成が可能になります。
# 3. **音声合成モデル**: TTS（Text-to-Speech）技術を用いて、テキストから自然な音声を生成します。これにより、音声アシスタントやナレーションの自動生成が可能です。
# 生成AIは、クリエイティブな分野だけでなく、ビジネス、教育、医療など多岐にわたる分野で活用されています。例えば、カスタマーサポートの自動化、教育コンテンツのパーソナライズ、医療データの解析などに応用されています。
# この技術は急速に進化しており、倫理的な課題やデータのプライバシーに関する議論も活発に行われています。生成AIを活用する際には、その利点とリスクを十分に理解し、適切に運用することが重要です。
