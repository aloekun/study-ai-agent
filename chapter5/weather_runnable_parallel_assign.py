from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
import pprint

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


retriever = TavilySearchAPIRetriever(k=3)

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | model
#     | StrOutputParser()
# )
# chain = {
#     "question": RunnablePassthrough(),
#     "context": retriever,
# } | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())
# 上2つの書き方と下の書き方は同じ挙動になる
# プロンプトに渡す前に RunnablePassthrough で受け取ることで、プロンプトの変数に結果を渡すことができる
# RannableParallel を作った後に assign で結果を付け足すこともできる
chain = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
).assign(answer=prompt | model | StrOutputParser())

output = chain.invoke("ワシントンD.C.の今日の天気は？")
pprint.pprint(output)

# 出力例
# {'answer': '文脈には具体的な今日の天気の情報は含まれていませんが、ワシントンD.C.の今日の天気予報については、The Weather '
#            'ChannelやAccuWeather、ウェザーニュースのサイトで確認できると記載されています。具体的な天気の詳細はそれらのリンクを参照してください。',
#  'context': [Document(metadata={'title': 'ワシントン, DC, アメリカ合衆国の天気予報と天候状況 - The Weather Channel | Weather.com', 'source': 'https://weather.com/ja-JP/weather/today/l/Washington+DC+United+States?canonicalCityId=5449cc9af33d6584872016be78d0340d', 'score': 0.59655124, 'images': []}, page_content='The Weather ChannelとWeather.comによる今日と今夜のワシントン, DC, アメリカ合衆国の天気予報、天候状況、ドップラーレーダー'),
#              Document(metadata={'title': 'ワシントンD.C., DCの3日間の天気予報 | AccuWeather', 'source': 'https://www.accuweather.com/ja/us/washington/20006/weather-forecast/327659', 'score': 0.57064295, 'images': []}, page_content='ワシントンD.C., DC Weather Forecast, with current conditions, wind, air quality, and what to expect for the next 3 days. 戻る ワシントンD.C., コロンビア特別区'),
#              Document(metadata={'title': 'ワシントンd.c.の天気予報 - ウェザーニュース', 'source': 'https://weathernews.jp/onebox/tenki/world/7/us/washington-dc/', 'score': 0.564027, 'images': []}, page_content='【ワシントンD.C.の2週間先までの天気予報】ワシントンD.C.の最高気温と最低気温を比較して旅行や出張に備えよう。雨雲レーダーは48時間先、国内外の空港の天気も確認できます。世界の天気に関するサポートは【ウェザーニュース】にお任せください。')],
#  'question': 'ワシントンD.C.の今日の天気は？'}
