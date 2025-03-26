from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した動物を説明してください。"),
        ("human", "{animal}"),
    ]
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

chain = prompt | model | output_parser
# streamメソッドを使って、ストリーム処理を行う
# streamメソッドは、ジェネレーターを返すので、for文で順次出力する
for chunk in chain.stream({"animal": "猫"}):
    print(chunk, end="", flush=True)

# 出力例（streamなので、以下の文章が随時追加される様子が見える）
# 猫（ねこ）は、哺乳類の一種で、ネコ科に属する動物です。一般的には家庭で飼われるペットとして知られていますが、野生の猫も存在します。猫は小型でしなやかな体を持ち、鋭い爪と優れた運動能力を備えています。
# 猫の特徴として、柔らかい毛皮、鋭い感覚（特に視覚と聴覚）、そして独特の鳴き声（「ニャー」と鳴く）があります。性格は一般的に独立心が強く、好奇心旺盛ですが、飼い主に対して愛情を示すこともあります。
# 猫は肉食性で、主に小動物や鳥を捕食します。狩猟本能が強く、遊びの中でその本能を発揮することが多いです。また、猫は非常に清潔好きで、自分の体を舐めて毛づくろいをする習性があります。
# 猫は多様な品種が存在し、それぞれに異なる外見や性格があります。一般的な品種には、ペルシャ猫、アメリカンショートヘア、シャム猫などがあります。猫はその愛らしさや個性的な性格から、多くの人々に愛されている動物です。
