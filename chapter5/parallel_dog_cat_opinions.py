from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import pprint
from langchain_core.runnables import RunnableParallel

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

# RunnableParallelを使って、複数のチェーンを並列に実行する
parallel_chain = RunnableParallel(
    {
        "dog_opinion": dog_chain,
        "cat_opinion": cat_chain,
    }
)

# 並列実行した結果を順番に出力
output = parallel_chain.invoke({"topic": "人間にはどんな性格の人がいるか"})
pprint.pprint(output)

# 出力例
# {'cat_opinion': '人間にはさまざまな性格の人がいますよね。例えば、社交的で外向的な人もいれば、内向的で静かな人もいます。猫も同じように、性格がそれぞれ異なります。例えば、甘えん坊で人懐っこい猫もいれば、ちょっとクールで独立心の強い猫もいます。\n'
#                 '\n'
#                 '性格の違いは、猫と人間の両方において、個性を形成する大切な要素です。猫の性格を理解することで、彼らとのコミュニケーションがより深まりますし、人間同士でもお互いの性格を理解することで、より良い関係を築くことができると思います。あなたはどんな性格の人が好きですか？それに似た猫の性格を考えてみるのも面白いかもしれませんね！',
#  'dog_opinion': '人間にはさまざまな性格の人がいますよね。例えば、社交的で明るい人、内向的で静かな人、思慮深くて慎重な人、冒険心旺盛で行動的な人など、多様な性格が存在します。\n'
#                 '\n'
#                 '犬も同様に、性格は犬種や個体によって異なります。例えば、ラブラドール・レトリーバーはフレンドリーで社交的な性格が多いですが、シベリアン・ハスキーは独立心が強く、少し頑固なところがあります。人間の性格と同じように、犬もそれぞれの個性を持っているので、どんな性格の犬が自分に合うかを考えるのも楽しいですね。あなたはどんな性格の犬が好きですか？'}
