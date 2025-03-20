from openai import OpenAI

client = OpenAI()

image_url = "https://photock.jp/photo/big_webp/photo0000-4183.webp"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user", "content": [
                {"type": "text", "text": "画像を説明してください。"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ],
        },
    ],
)
print(response.choices[0].message.content)

# 出力例
# 画像には、砂の上で丸まって眠っている小さな動物が写っています。毛は柔らかく、淡い茶色をしています。大きな耳が特徴的で、目を閉じてリラックスしている様子が見受けられます。周囲は沙漠のようで、背景は明るい砂の色です。この動物は、通常、乾燥地に住む獣で、小さくてもかわいらしい姿が印象的です。
