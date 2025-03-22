from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# レシピの構造を決めるクラス
class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")


# Recipeクラスを解析して出力形式を作成
output_parser = PydanticOutputParser(pydantic_object=Recipe)

# 入力するプロンプトを作成
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。\n\n{format_instructions}"),
        ("human", "{dish}"),
    ]
)

# プロンプトと出力形式を組み合わせる
prompt_with_format_instructions = prompt.partial(
    format_instructions=output_parser.get_format_instructions()
)

# OpenAIのモデルを作成
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
    response_format={"type": "json_object"}
)

# プロンプト、モデル、出力解析器を組み合わせてChainを作成
chain = prompt_with_format_instructions | model | output_parser

# 入力を与えてChainを実行
recipe = chain.invoke({"dish": "酢豚"})
print(type(recipe))
print(recipe)

# 出力例
# <class '__main__.Recipe'>
# ingredients=['豚肉 (肩ロースまたはバラ肉) 300g', '玉ねぎ 1個', 'ピーマン 1個', 'パプリカ 1個', 'にんじん 1本', '片栗粉 適量', 'サラダ油 適量', '酢 50ml', '砂糖 30g', '醤油 30ml', 'ケチャップ 50g', '鶏がらスープの素 小さじ1', '塩 適量', '胡椒 適量'] steps=['豚肉を一口大に切り、塩と胡椒で下味をつける。', '玉ねぎ、ピーマン、パプリカ、にんじんを食べやすい大きさに切る。', '豚肉に片栗粉をまぶし、熱したサラダ油で揚げる。', '別の鍋に酢、砂糖、醤油、ケチャップ、鶏がらスープの素を入れて混ぜ、煮立たせる。', '揚げた豚肉と野菜を鍋に加え、全体をよく混ぜ合わせる。', '味を見て、必要に応じて塩や胡椒で調整する。', '皿に盛り付けて完成。'