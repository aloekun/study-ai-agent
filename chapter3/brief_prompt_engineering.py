from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "質問への回答は100文字程度で答えてください。"},
        {"role": "user", "content": "プロンプトエンジニアリングとは"},
    ],
)
print(response.choices[0].message.content)

# 出力例
# プロンプトエンジニアリングとは、AIモデルに対して適切な入力（プロンプト）を設計し、望ましい出力を引き出す技術や手法のことです。これによりAIのパフォーマンスを向上させることができます。
