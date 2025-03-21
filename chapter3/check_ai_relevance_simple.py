from openai import OpenAI

client = OpenAI()

# Fiew-shotプロンプティングを使ったAIの関連性チェック
# Few-shotプロンプティング：少量の情報を与えて、結果を得る手法
# とりあえずシステム条件を与えているが、あまり有効活用してもらえない例
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "入力がAIに関係するか回答してください。"},
        {"role": "user", "content": "レタスとキャベツは似ている。"},
    ],
)
print(response.choices[0].message.content)

# 出力例
# はい、レタスとキャベツはどちらも葉物野菜で、見た目や食感が似ていますが、いくつかの違いもあります。レタスは一般的に柔らかく、水分が多いのが特徴です。一方、キャベツは葉が厚くて crunchy で、さまざまな料理に使われます。この2つは、サラダや料理の素材としてよく利用されますが、それぞれの特性に応じて使い分けられます。
