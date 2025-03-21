from openai import OpenAI

client = OpenAI()

# Few-shotプロンプティングを使ったAIの関連性チェック
# Few-shotプロンプティング：少量の情報を与えて、結果を得る手法
# 出力例を複数見せることで、結果のフォーマットを制限するサンプル
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "入力がAIに関係するか回答してください。"},
        {"role": "user", "content": "近年のAIはマルチモーダルに対応して、画像や音声を認識できる"},
        {"role": "assistant", "content": "true"},
        {"role": "user", "content": "プログラマがプログラムを書く機会は減るだろう。なぜならばAIが台頭してくるからだ。"},
        {"role": "assistant", "content": "true"},
        {"role": "user", "content": "公園で鳩が羽ばたいている。"},
        {"role": "assistant", "content": "false"},
        {"role": "user", "content": "北極星とベガは同じぐらい輝いている。"},
    ],
)
print(response.choices[0].message.content)

# 出力例
# false
