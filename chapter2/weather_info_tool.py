import json
from openai import OpenAI


def get_current_weather(location, unit="fahrenheit"):
    if "kanagawa" in location.lower():
        return json.dumps({"location": "Kanagawa", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


# ツールの定義
# function["parameters"]["properties"]["location"]["description"]に "which name is English" を書かないと日本語で返してきてコケたので、調整した
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, which name is English, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

client = OpenAI()

messages = [
    {"role": "user", "content": "神奈川の天気はどうですか？"},
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)
# print(response.to_json(indent=2))

response_message = response.choices[0].message
messages.append(response_message.to_dict())

available_functions = {
    "get_current_weather": get_current_weather,
}

# 使いたい関数は複数あるかもしれないのでループ
for tool_call in response_message.tool_calls:
    # 関数を実行
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )

    # 関数の実行結果を会話履歴としてmessagesに追加
    messages.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }
    )

json_output = json.dumps(messages, ensure_ascii=False, indent=2)
print(json_output)

# 出力例
# [
#   {
#     "role": "user",
#     "content": "神奈川の天気はどうですか？"
#   },
#   {
#     "content": null,
#     "refusal": null,
#     "role": "assistant",
#     "tool_calls": [
#       {
#         "id": "call_1ZZAnwVxq88tu9EdzbMBhhl1",
#         "function": {
#           "arguments": "{\"location\":\"Kanagawa, Japan\",\"unit\":\"celsius\"}",
#           "name": "get_current_weather"
#         },
#         "type": "function"
#       }
#     ],
#     "annotations": []
#   },
#   {
#     "tool_call_id": "call_1ZZAnwVxq88tu9EdzbMBhhl1",
#     "role": "tool",
#     "name": "get_current_weather",
#     "content": "{\"location\": \"Kanagawa\", \"temperature\": \"10\", \"unit\": \"celsius\"}"
#   }
# ]
