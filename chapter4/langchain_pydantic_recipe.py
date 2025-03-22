from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")


output_parser = PydanticOutputParser(pydantic_object=Recipe)

format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

# 出力例(Recipeクラスのデータ構造の出力フォーマットを自動で作成)
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
#
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
#
# Here is the output schema:
# ```
# {"properties": {"ingredients": {"description": "ingredients of the dish", "items": {"type": "string"}, "title": "Ingredients", "type": "array"}, "steps": {"description": "steps to make the dish", "items": {"type": "string"}, "title": "Steps", "type": "array"}}, "required": ["ingredients", "steps"]}


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "ユーザーが入力した料理のレシピを考えてください。\n\n"
            "{format_instructions}",
        ),
        ("human", "{dish}"),
    ],
)

prompt_with_format_instructions = prompt.partial(
    format_instructions=format_instructions
)

prompt_value = prompt_with_format_instructions.invoke({"dish": "小籠包"})
# print("=== role: system ===")
# print(prompt_value.messages[0].content)
# print("=== role: user ===")
# print(prompt_value.messages[1].content)

# 出力例（出力フォーマットとユーザーの入力を組み合わせてプロンプトを作成）
# === role: system ===
# ユーザーが入力した料理のレシピを考えてください。
#
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
#
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
#
# Here is the output schema:
# ```
# {"properties": {"ingredients": {"description": "ingredients of the dish", "items": {"type": "string"}, "title": "Ingredients", "type": "array"}, "steps": {"description": "steps to make the dish", "items": {"type": "string"}, "title": "Steps", "type": "array"}}, "required": ["ingredients", "steps"]}
# ```
# === role: user ===
# 小籠包


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ai_message = model.invoke(prompt_value)
print(ai_message.content)

# 出力例（AIモデルによるレシピの生成、材料と手順を出力した）
# {
#   "ingredients": [
#     "小麦粉 300g",
#     "水 150ml",
#     "豚ひき肉 200g",
#     "生姜 1片",
#     "ネギ 1本",
#     "醤油 大さじ2",
#     "ごま油 大さじ1",
#     "塩 小さじ1",
#     "こしょう 少々",
#     "鶏ガラスープ 100ml",
#     "ゼラチン 5g"
#   ],
#   "steps": [
#     "小麦粉と水を混ぜて、こねて生地を作り、30分休ませる。",
#     "生姜とネギをみじん切りにする。",
#     "豚ひき肉に生姜、ネギ、醤油、ごま油、塩、こしょうを加えてよく混ぜる。",
#     "鶏ガラスープを加え、さらに混ぜて肉だねを作る。",
#     "ゼラチンを水でふやかし、肉だねに加えてよく混ぜる。",
#     "生地を小さく分けて、薄く伸ばす。",
#     "生地の中央に肉だねをのせ、包み込む。",
#     "蒸し器で約15分蒸す。",
#     "熱々の小籠包をお皿に盛り付けて完成。"
#   ]
# }
