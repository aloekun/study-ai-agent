from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": '人物一覧を次のJSON形式で出力してください。\n{"people": ["aaa","bbb"]}'},
        {"role": "user", "content": "桃太郎はおじいさんとおばあさんからきびだんごをもらいました。"},
    ],
    response_format={"type": "json_object"},
)
print(response.choices[0].message.content)

# 出力例
# {"people": ["桃太郎","おじいさん","おばあさん"]}
