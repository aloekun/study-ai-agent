from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")


prompt = ChatPromptTemplate(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini")

# with_structured_output を使ってChainすることで、 PydanticOutputParse を使わずとも出力できる
# 構造データの出力はこちらの書き方がお手軽
# ただし、すべてのモデルで with_structured_output が使えるわけではないので注意
chain = prompt | model.with_structured_output(Recipe)

recipe = chain.invoke({"dish": "サーターアンダギー"})
print(type(recipe))
print(recipe)

# 出力例
# <class '__main__.Recipe'>
# ingredients=['薄力粉 200g', '砂糖 100g', '卵 2個', '牛乳 50ml', 'ベーキングパウダー 小さじ1', '塩 ひとつまみ', 'サラダ油 適量（揚げ用）'] steps=['ボウルに薄力粉、砂糖、ベーキングパウダー、塩を入れて混ぜます。', '別のボウルで卵をよく溶き、牛乳を加えて混ぜます。', '卵と牛乳の混ぜたものを粉類のボウルに加え、全体が混ざるまでよく混ぜます。', '生地がまとまったら、手で小さなボール状に成形します。', '鍋にサラダ油を入れ、170℃に熱します。', '成形した生地を油に入れ、きつね色になるまで揚げます。', '揚げたサーターアンダギーを油から取り出し、キッチンペーパーの上で余分な油を切ります。', '温かい状態でお召し上がりください。']
