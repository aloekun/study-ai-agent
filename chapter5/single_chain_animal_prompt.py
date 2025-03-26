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

# prompt_value = prompt.invoke({"animal": "犬"})
# ai_message = model.invoke(prompt_value)
# output = output_parser.invoke(ai_message)
# 上の 3 行の invoke と、下の Chain して一括invoke する処理は、同じ挙動になる
chain = prompt | model | output_parser
output = chain.invoke({"animal": "犬"})

print(output)

# 出力例
# 犬は、家畜化された哺乳類で、イヌ科に属する動物です。人間との関係が非常に深く、古代から伴侶動物として飼われてきました。犬は多様な品種が存在し、それぞれ異なる外見や性格、能力を持っています。
# 犬は非常に社交的で、忠誠心が強いことで知られています。飼い主との絆を深めるために、愛情を示したり、遊んだりすることが好きです。また、嗅覚が非常に優れており、警察犬や救助犬、盲導犬など、さまざまな役割を果たすことができます。
# 犬はまた、運動や遊びを通じて健康を維持することが重要で、散歩やトレーニングを通じて飼い主と一緒に過ごす時間を楽しむことができます。犬は「人間の最良の友」とも呼ばれ、その存在は多くの人々に喜びや癒しをもたらしています。
