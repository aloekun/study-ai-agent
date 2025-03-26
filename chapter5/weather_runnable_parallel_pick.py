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

# assign で結果を付け足すこともできるし、pickで結果をフィルタすることもできる
chain = (
    RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "context": retriever,
        }
    )
    .assign(answer=prompt | model | StrOutputParser())
    .pick(["context", "answer"])
)

output = chain.invoke("ボストンの今日の天気は？")
pprint.pprint(output)

# 出力例
# {'answer': '文脈にはボストンの今日の天気に関する具体的な情報は含まれていません。ただし、ボストンの今日の天気や週間天気、過去天気についての情報が掲載されていることが示されています。詳細な天気情報を知りたい場合は、提供されたリンクを参照してください。',
#  'context': [Document(metadata={'title': 'ボストン(アメリカ)の天気 - 日本気象協会 tenki.jp', 'source': 'https://tenki.jp/world/7/92/72509/', 'score': 0.7574541, 'images': []}, page_content='ボストン(アメリカ)の今日の天気・週間天気、実況天気に加え、7日前までの過去天気を掲載しています。雲量や視程、気圧、湿度など、どこより'),
#              Document(metadata={'title': 'ボストンの天気予報 - ウェザーニュース', 'source': 'https://weathernews.jp/onebox/tenki/world/7/us/boston/', 'score': 0.534443, 'images': []}, page_content='【ボストンの2週間先までの天気予報】ボストンの最高気温と最低気温を比較して旅行や出張に備えよう。雨雲レーダーは48時間先、国内外の空港の天気も確認できます。世界の天気に関するサポートは【ウェザーニュース】にお任せください。'),
#              Document(metadata={'title': 'ボストン, MAの現在の天気概況 - AccuWeather', 'source': 'https://www.accuweather.com/ja/us/boston/02108/current-weather/348735', 'score': 0.5315261, 'images': []}, page_content='翌日の備えに役立ててください。レーダーや1時間ごとの予報、さらには最大1分単位の予報を使用して、ボストン, maの気象条件を1日先に確認し')]}
