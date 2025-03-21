from openai import OpenAI

client = OpenAI()

# Zero-shot プロンプティングを使った感情分類
# Zero-shot プロンプティング：質問以外の情報を与えずに、結果を得る手法
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "入力をポジティブ・ネガティブ・中立のどれかに分類してください。",
        },
        {
            "role": "user",
            "content": "雨の日は靴が濡れるので、げんなりする。",
        },
    ],
)
print(response.choices[0].message.content)

# 出力例
# ネガティブ
