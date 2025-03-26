from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的AIです。2つの意見をまとめてください。"),
        ("human", "犬寄りの意見: {dog_opinion}\n猫寄りの意見: {cat_opinion}"),
    ]
)

# RunnableParallelを使って、複数のチェーンを並列に実行する
# その後にChainで意見をまとめて出力する
synthesize_chain = (
    RunnableParallel(
        {
            "dog_opinion": dog_chain,
            "cat_opinion": cat_chain,
        }
    )
    | synthesize_prompt
    | model
    | output_parser
)

output = synthesize_chain.invoke({"topic": "クジラの性格"})
print(output)

# 出力例
# 犬寄りの意見と猫寄りの意見をまとめると、クジラは非常に社交的で知能が高く、家族や群れを大切にする性格を持つ生き物であるという点で共通しています。犬と猫の両方が、飼い主や他の動物との絆を重視し、それぞれの個性を持っていることも強調されています。犬は群れでの生活や感受性の高さが魅力とされ、猫は独立心や愛らしさが特別な魅力として挙げられています。どちらの意見も、クジラと犬、猫の関係性を通じて、私たち人間にとって彼らが大切な存在であることを示しています。
