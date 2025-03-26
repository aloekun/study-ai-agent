from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

dog_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは犬派です。ユーザーの入力に対して犬の情報を絡めつつ意見をください。"),
        ("human", "{topic}"),
    ]
)
dog_chain = dog_prompt | model | output_parser

cat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは猫派です。ユーザーの入力に対して猫の情報を絡めつつ意見をください。"),
        ("human", "{topic}"),
    ]
)
cat_chain = cat_prompt | model | output_parser

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的AIです。{topic}について2つの意見をまとめてください。"),
        ("human", "犬寄りの意見: {dog_opinion}\n猫寄りの意見: {cat_opinion}"),
    ]
)

# itemgetter("topic")を使って、入力の中から"topic"キーの値を取得する
synthesize_chain = (
    {
        "dog_opinion": dog_chain,
        "cat_opinion": cat_chain,
        "topic": itemgetter("topic"),
    }
    | synthesize_prompt
    | model
    | output_parser
)

output = synthesize_chain.invoke({"topic": "ゾウの性格"})
print(output)

# 出力例
# ゾウの性格についての意見をまとめると、以下のようになります。
# 1. **犬寄りの意見**: ゾウは非常に知能が高く、社交的な動物であり、強い絆を持つことが特徴です。彼らは家族や群れを大切にし、感情を豊かに表現します。この点で、犬と共通する部分が多く、犬もまた社交的で人間との絆を深めることが得意です。どちらの動物も愛情や絆を重視し、喜びや悲しみを表現する姿が魅力的です。
# 2. **猫寄りの意見**: ゾウは知能が高く、社交的で家族を大切にする性格を持っていますが、猫も同様に独立心が強く、個々の性格が異なる魅力を持っています。猫は飼い主との絆を深めることができ、愛情を示すときには甘えん坊になることがあります。ゾウと猫はそれぞれ異なる性格を持ちながらも、私たちに喜びを与えてくれる点で共通しています。
# このように、ゾウは犬や猫と同様に、知能や社交性、感情表現において多様な魅力を持つ動物であることがわかります。
