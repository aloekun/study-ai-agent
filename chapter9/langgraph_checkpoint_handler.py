from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
import operator
from pprint import pprint
from pydantic import BaseModel, Field
from typing import Annotated, Any


# グラフのステートを定義
class State(BaseModel):
    query: str
    messages: Annotated[list[BaseModel], operator.add] = Field(default=[])


# メッセージを追加するノード関数
def add_message(state: State) -> dict[str, Any]:
    additional_messages = []
    if not state.messages:
        additional_messages.append(
            SystemMessage(content="あなたは最小限の応答をする対話エージェントです。")
        )
    additional_messages.append(HumanMessage(content=state.query))
    return {"messages": additional_messages}


# LLM からの応答を追加するノード関数
def llm_response(state: State) -> dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    ai_message = llm.invoke(state.messages)
    return {"messages": [ai_message]}


def print_checkpoint_dump(checkpointer: BaseCheckpointSaver, config: RunnableConfig):
    checkpoint_tuple = checkpointer.get_tuple(config)

    print("チェックポイントデータ:")
    pprint(checkpoint_tuple.checkpoint)
    print("\nメタデータ:")
    pprint(checkpoint_tuple.metadata)


# グラフを設定
graph = StateGraph(State)
graph.add_node("add_message", add_message)
graph.add_node("llm_response", llm_response)

graph.set_entry_point("add_message")
graph.add_edge("add_message", "llm_response")
graph.add_edge("llm_response", END)

# チェックポインターを設定
checkpointer = MemorySaver()

# グラフをコンパイル
compiled_graph = graph.compile(checkpointer=checkpointer)

# 使用例

# thuread_id を指定して、やり取りを開始する
config = {"configurable": {"thread_id": "example-1"}}
user_query = State(query="私の好きなものはビリヤニです。覚えておいてね。")
first_response = compiled_graph.invoke(user_query, config)
first_response

# {'query': '私の好きなものはビリヤニです。覚えておいてね。',
#  'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}),
#   HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}),
#   AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})]}

for checkpoint in checkpointer.list(config):
    print(checkpoint)

# 4 行目のチェックポイントで、「'__start__': State(query='私の好きなものはビリヤニです。覚えておいてね。', messages=[])」が設定されている
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-bcae-6dea-8002-910018423b39'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:03:02.971521+00:00', 'id': '1f010945-bcae-6dea-8002-910018423b39', 'channel_values': {'query': '私の好きなものはビリヤニです。覚えておいてね。', 'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}), AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})], 'llm_response': 'llm_response'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.0.7431413905258895', 'query': '00000000000000000000000000000002.0.4687477697567908', 'messages': '00000000000000000000000000000004.0.39127648887730593', 'start:add_message': '00000000000000000000000000000003.0.19306014814639993', 'add_message': '00000000000000000000000000000004.0.40550858776425647', 'llm_response': '00000000000000000000000000000004.0.5215365487575174'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}, 'add_message': {'start:add_message': '00000000000000000000000000000002.0.017773432446775095'}, 'llm_response': {'add_message': '00000000000000000000000000000003.0.8086597561490658'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': {'llm_response': {'messages': [AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})]}}, 'step': 2, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac72-6b99-8001-8b51d8b20c3b'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac72-6b99-8001-8b51d8b20c3b'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:03:01.269168+00:00', 'id': '1f010945-ac72-6b99-8001-8b51d8b20c3b', 'channel_values': {'query': '私の好きなものはビリヤニです。覚えておいてね。', 'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={})], 'add_message': 'add_message'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.0.7431413905258895', 'query': '00000000000000000000000000000002.0.4687477697567908', 'messages': '00000000000000000000000000000003.0.4966113972630878', 'start:add_message': '00000000000000000000000000000003.0.19306014814639993', 'add_message': '00000000000000000000000000000003.0.8086597561490658'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}, 'add_message': {'start:add_message': '00000000000000000000000000000002.0.017773432446775095'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': {'add_message': {'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={})]}}, 'step': 1, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac67-6b92-8000-974712661a3e'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac67-6b92-8000-974712661a3e'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:03:01.264656+00:00', 'id': '1f010945-ac67-6b92-8000-974712661a3e', 'channel_values': {'query': '私の好きなものはビリヤニです。覚えておいてね。', 'messages': [], 'start:add_message': '__start__'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.0.7431413905258895', 'query': '00000000000000000000000000000002.0.4687477697567908', 'messages': '00000000000000000000000000000002.0.7459668739743379', 'start:add_message': '00000000000000000000000000000002.0.017773432446775095'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac5d-60ed-bfff-039e29a522a9'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac5d-60ed-bfff-039e29a522a9'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:03:01.260299+00:00', 'id': '1f010945-ac5d-60ed-bfff-039e29a522a9', 'channel_values': {'__start__': State(query='私の好きなものはビリヤニです。覚えておいてね。', messages=[])}, 'channel_versions': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}, 'versions_seen': {'__input__': {}}, 'pending_sends': []}, metadata={'source': 'input', 'writes': {'__start__': State(query='私の好きなものはビリヤニです。覚えておいてね。', messages=[])}, 'step': -1, 'parents': {}}, parent_config=None, pending_writes=[])

print_checkpoint_dump(checkpointer, config)

# AI メッセージの内容を確認すると、次のように好きなものを理解したことがわかる
# 'llm_response': {'messages': [AIMessage(content='了解しました。ビリヤニが好きなんですね。', ...

# チェックポイントデータ:
# {'channel_values': {'llm_response': 'llm_response',
#                     'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}),
#                                  HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}),
#                                  AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})],
#                     'query': '私の好きなものはビリヤニです。覚えておいてね。'},
#  'channel_versions': {'__start__': '00000000000000000000000000000002.0.7431413905258895',
#                       'add_message': '00000000000000000000000000000004.0.40550858776425647',
#                       'llm_response': '00000000000000000000000000000004.0.5215365487575174',
#                       'messages': '00000000000000000000000000000004.0.39127648887730593',
#                       'query': '00000000000000000000000000000002.0.4687477697567908',
#                       'start:add_message': '00000000000000000000000000000003.0.19306014814639993'},
#  'id': '1f010945-bcae-6dea-8002-910018423b39',
#  'pending_sends': [],
#  'ts': '2025-04-03T14:03:02.971521+00:00',
#  'v': 1,
#  'versions_seen': {'__input__': {},
#                    '__start__': {'__start__': '00000000000000000000000000000001.0.3799340651358638'},
#                    'add_message': {'start:add_message': '00000000000000000000000000000002.0.017773432446775095'},
#                    'llm_response': {'add_message': '00000000000000000000000000000003.0.8086597561490658'}}}

# メタデータ:
# {'parents': {},
#  'source': 'loop',
#  'step': 2,
#  'writes': {'llm_response': {'messages': [AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})]}}}

# 同じスレッドIDのまま、質問してみる
user_query = State(query="私の好物は何か覚えてる？")
second_response = compiled_graph.invoke(user_query, config)
second_response

# すると、先に教えた内容を覚えていて、正しい回答をしてくれる
# {'query': '私の好物は何か覚えてる？',
#  'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}),
#   HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}),
#   AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60}),
#   HumanMessage(content='私の好物は何か覚えてる？', additional_kwargs={}, response_metadata={}),
#   AIMessage(content='はい、ビリヤニが好きです。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 78, 'total_tokens': 89, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-09675a8a-a51c-4010-80e6-504a39f73f72-0', usage_metadata={'input_tokens': 78, 'output_tokens': 11, 'total_tokens': 89})]}

for checkpoint in checkpointer.list(config):
    print(checkpoint)

# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f01096f-d4bf-68dd-8006-ec13bc9629d7'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:21:52.923840+00:00', 'id': '1f01096f-d4bf-68dd-8006-ec13bc9629d7', 'channel_values': {'query': '私の好物は何か覚えてる？', 'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}), AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60}), HumanMessage(content='私の好物は何か覚えてる？', additional_kwargs={}, response_metadata={}), AIMessage(content='はい、ビリヤニが好きです。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 78, 'total_tokens': 89, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-09675a8a-a51c-4010-80e6-504a39f73f72-0', usage_metadata={'input_tokens': 78, 'output_tokens': 11, 'total_tokens': 89})], 'llm_response': 'llm_response'}, 'channel_versions': {'__start__': '00000000000000000000000000000006.0.5596990217269375', 'query': '00000000000000000000000000000006.0.23669002458305777', 'messages': '00000000000000000000000000000008.0.3610030902061686', 'start:add_message': '00000000000000000000000000000007.0.867225098934374', 'add_message': '00000000000000000000000000000008.0.4381830318051795', 'llm_response': '00000000000000000000000000000008.0.706855128848778'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000005.0.49697399127595143'}, 'add_message': {'start:add_message': '00000000000000000000000000000006.0.595740781639936'}, 'llm_response': {'add_message': '00000000000000000000000000000007.0.11707687097824016'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': {'llm_response': {'messages': [AIMessage(content='はい、ビリヤニが好きです。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 78, 'total_tokens': 89, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-09675a8a-a51c-4010-80e6-504a39f73f72-0', usage_metadata={'input_tokens': 78, 'output_tokens': 11, 'total_tokens': 89})]}}, 'step': 6, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f01096f-c5e4-6fa0-8005-d47726a9f5e6'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f01096f-c5e4-6fa0-8005-d47726a9f5e6'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:21:51.366308+00:00', 'id': '1f01096f-c5e4-6fa0-8005-d47726a9f5e6', 'channel_values': {'query': '私の好物は何か覚えてる？', 'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}), AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60}), HumanMessage(content='私の好物は何か覚えてる？', additional_kwargs={}, response_metadata={})], 'add_message': 'add_message'}, 'channel_versions': {'__start__': '00000000000000000000000000000006.0.5596990217269375', 'query': '00000000000000000000000000000006.0.23669002458305777', 'messages': '00000000000000000000000000000007.0.9956692739435848', 'start:add_message': '00000000000000000000000000000007.0.867225098934374', 'add_message': '00000000000000000000000000000007.0.11707687097824016', 'llm_response': '00000000000000000000000000000005.0.24949122672547686'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000005.0.49697399127595143'}, 'add_message': {'start:add_message': '00000000000000000000000000000006.0.595740781639936'}, 'llm_response': {'add_message': '00000000000000000000000000000003.0.8086597561490658'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': {'add_message': {'messages': [HumanMessage(content='私の好物は何か覚えてる？', additional_kwargs={}, response_metadata={})]}}, 'step': 5, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f01096f-c5c6-69ea-8004-d8ae9ddc1641'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f01096f-c5c6-69ea-8004-d8ae9ddc1641'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:21:51.353862+00:00', 'id': '1f01096f-c5c6-69ea-8004-d8ae9ddc1641', 'channel_values': {'query': '私の好物は何か覚えてる？', 'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}), AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})], 'start:add_message': '__start__'}, 'channel_versions': {'__start__': '00000000000000000000000000000006.0.5596990217269375', 'query': '00000000000000000000000000000006.0.23669002458305777', 'messages': '00000000000000000000000000000006.0.6258542382533576', 'start:add_message': '00000000000000000000000000000006.0.595740781639936', 'add_message': '00000000000000000000000000000004.0.40550858776425647', 'llm_response': '00000000000000000000000000000005.0.24949122672547686'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000005.0.49697399127595143'}, 'add_message': {'start:add_message': '00000000000000000000000000000002.0.017773432446775095'}, 'llm_response': {'add_message': '00000000000000000000000000000003.0.8086597561490658'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': None, 'step': 4, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f01096f-c5a3-6eaf-8003-6f87d952a245'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f01096f-c5a3-6eaf-8003-6f87d952a245'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:21:51.339652+00:00', 'id': '1f01096f-c5a3-6eaf-8003-6f87d952a245', 'channel_values': {'query': '私の好きなものはビリヤニです。覚えておいてね。', 'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}), AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})], '__start__': State(query='私の好物は何か覚えてる？', messages=[])}, 'channel_versions': {'__start__': '00000000000000000000000000000005.0.49697399127595143', 'query': '00000000000000000000000000000002.0.4687477697567908', 'messages': '00000000000000000000000000000004.0.39127648887730593', 'start:add_message': '00000000000000000000000000000003.0.19306014814639993', 'add_message': '00000000000000000000000000000004.0.40550858776425647', 'llm_response': '00000000000000000000000000000005.0.24949122672547686'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}, 'add_message': {'start:add_message': '00000000000000000000000000000002.0.017773432446775095'}, 'llm_response': {'add_message': '00000000000000000000000000000003.0.8086597561490658'}}, 'pending_sends': []}, metadata={'source': 'input', 'writes': {'__start__': State(query='私の好物は何か覚えてる？', messages=[])}, 'step': 3, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-bcae-6dea-8002-910018423b39'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-bcae-6dea-8002-910018423b39'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:03:02.971521+00:00', 'id': '1f010945-bcae-6dea-8002-910018423b39', 'channel_values': {'query': '私の好きなものはビリヤニです。覚えておいてね。', 'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}), AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})], 'llm_response': 'llm_response'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.0.7431413905258895', 'query': '00000000000000000000000000000002.0.4687477697567908', 'messages': '00000000000000000000000000000004.0.39127648887730593', 'start:add_message': '00000000000000000000000000000003.0.19306014814639993', 'add_message': '00000000000000000000000000000004.0.40550858776425647', 'llm_response': '00000000000000000000000000000004.0.5215365487575174'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}, 'add_message': {'start:add_message': '00000000000000000000000000000002.0.017773432446775095'}, 'llm_response': {'add_message': '00000000000000000000000000000003.0.8086597561490658'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': {'llm_response': {'messages': [AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60})]}}, 'step': 2, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac72-6b99-8001-8b51d8b20c3b'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac72-6b99-8001-8b51d8b20c3b'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:03:01.269168+00:00', 'id': '1f010945-ac72-6b99-8001-8b51d8b20c3b', 'channel_values': {'query': '私の好きなものはビリヤニです。覚えておいてね。', 'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={})], 'add_message': 'add_message'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.0.7431413905258895', 'query': '00000000000000000000000000000002.0.4687477697567908', 'messages': '00000000000000000000000000000003.0.4966113972630878', 'start:add_message': '00000000000000000000000000000003.0.19306014814639993', 'add_message': '00000000000000000000000000000003.0.8086597561490658'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}, 'add_message': {'start:add_message': '00000000000000000000000000000002.0.017773432446775095'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': {'add_message': {'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={})]}}, 'step': 1, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac67-6b92-8000-974712661a3e'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac67-6b92-8000-974712661a3e'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:03:01.264656+00:00', 'id': '1f010945-ac67-6b92-8000-974712661a3e', 'channel_values': {'query': '私の好きなものはビリヤニです。覚えておいてね。', 'messages': [], 'start:add_message': '__start__'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.0.7431413905258895', 'query': '00000000000000000000000000000002.0.4687477697567908', 'messages': '00000000000000000000000000000002.0.7459668739743379', 'start:add_message': '00000000000000000000000000000002.0.017773432446775095'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}}, parent_config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac5d-60ed-bfff-039e29a522a9'}}, pending_writes=[])
# CheckpointTuple(config={'configurable': {'thread_id': 'example-1', 'checkpoint_ns': '', 'checkpoint_id': '1f010945-ac5d-60ed-bfff-039e29a522a9'}}, checkpoint={'v': 1, 'ts': '2025-04-03T14:03:01.260299+00:00', 'id': '1f010945-ac5d-60ed-bfff-039e29a522a9', 'channel_values': {'__start__': State(query='私の好きなものはビリヤニです。覚えておいてね。', messages=[])}, 'channel_versions': {'__start__': '00000000000000000000000000000001.0.3799340651358638'}, 'versions_seen': {'__input__': {}}, 'pending_sends': []}, metadata={'source': 'input', 'writes': {'__start__': State(query='私の好きなものはビリヤニです。覚えておいてね。', messages=[])}, 'step': -1, 'parents': {}}, parent_config=None, pending_writes=[])

print_checkpoint_dump(checkpointer, config)

# チェックポイントデータ:
# {'channel_values': {'llm_response': 'llm_response',
#                     'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}),
#                                  HumanMessage(content='私の好きなものはビリヤニです。覚えておいてね。', additional_kwargs={}, response_metadata={}),
#                                  AIMessage(content='了解しました。ビリヤニが好きなんですね。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 47, 'total_tokens': 60, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-237e39a0-a691-4981-9a1e-972fba92040f-0', usage_metadata={'input_tokens': 47, 'output_tokens': 13, 'total_tokens': 60}),
#                                  HumanMessage(content='私の好物は何か覚えてる？', additional_kwargs={}, response_metadata={}),
#                                  AIMessage(content='はい、ビリヤニが好きです。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 78, 'total_tokens': 89, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-09675a8a-a51c-4010-80e6-504a39f73f72-0', usage_metadata={'input_tokens': 78, 'output_tokens': 11, 'total_tokens': 89})],
#                     'query': '私の好物は何か覚えてる？'},
#  'channel_versions': {'__start__': '00000000000000000000000000000006.0.5596990217269375',
#                       'add_message': '00000000000000000000000000000008.0.4381830318051795',
#                       'llm_response': '00000000000000000000000000000008.0.706855128848778',
#                       'messages': '00000000000000000000000000000008.0.3610030902061686',
#                       'query': '00000000000000000000000000000006.0.23669002458305777',
#                       'start:add_message': '00000000000000000000000000000007.0.867225098934374'},
#  'id': '1f01096f-d4bf-68dd-8006-ec13bc9629d7',
#  'pending_sends': [],
#  'ts': '2025-04-03T14:21:52.923840+00:00',
#  'v': 1,
#  'versions_seen': {'__input__': {},
#                    '__start__': {'__start__': '00000000000000000000000000000005.0.49697399127595143'},
#                    'add_message': {'start:add_message': '00000000000000000000000000000006.0.595740781639936'},
#                    'llm_response': {'add_message': '00000000000000000000000000000007.0.11707687097824016'}}}

# メタデータ:
# {'parents': {},
#  'source': 'loop',
#  'step': 6,
#  'writes': {'llm_response': {'messages': [AIMessage(content='はい、ビリヤニが好きです。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 78, 'total_tokens': 89, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-09675a8a-a51c-4010-80e6-504a39f73f72-0', usage_metadata={'input_tokens': 78, 'output_tokens': 11, 'total_tokens': 89})]}}}

config = {"configurable": {"thread_id": "example-2"}}
user_query = State(query="私の好物は何ですか？")
other_thread_response = compiled_graph.invoke(user_query, config)
other_thread_response

# {'query': '私の好物は何ですか？',
#  'messages': [SystemMessage(content='あなたは最小限の応答をする対話エージェントです。', additional_kwargs={}, response_metadata={}),
#   HumanMessage(content='私の好物は何ですか？', additional_kwargs={}, response_metadata={}),
#   AIMessage(content='わかりません。あなたの好物は何ですか？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 38, 'total_tokens': 53, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b376dfbbd5', 'finish_reason': 'stop', 'logprobs': None}, id='run-77b961c8-5b29-462c-9893-1ddfe5ee8e56-0', usage_metadata={'input_tokens': 38, 'output_tokens': 15, 'total_tokens': 53})]}
