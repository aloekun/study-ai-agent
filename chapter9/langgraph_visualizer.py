from IPython.display import Image
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import Any


class State(BaseModel):
    query: str = Field(
        ..., description="ユーザーからの質問"
    )
    current_role: str = Field(
        default="", description="選定された回答ロール"
    )


def question_node(state: State) -> dict[str, Any]:
    query = state.query
    prompt = ChatPromptTemplate.from_template("""
質問に適切な回答をしてください。

質問: {query}
""".strip()
    )
    # 選択肢の番号のみを返すことを期待したいため、max_tokens の値を 1 に変更
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"query": query})

    return {"answer": answer}


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

workflow = StateGraph(State)

workflow.add_node("question", question_node)
workflow.add_node("end", END)

workflow.add_edge("question", "end")

# compiled は langgraph_role_workflow.py のような、 LangGraphでコンパイルされたグラフを渡す
compiled = workflow.compile()

Image(compiled.get_graph().draw_png())

# 出力例
# langgraph_role_workflow.png
