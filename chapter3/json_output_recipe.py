from openai import OpenAI

client = OpenAI()

# json形式で出力させるプロンプト
system_prompt = """\
ユーザーが入力sた料理のレシピを考えてください。
出力は以下のJSON形式にしてください。

'''
{
  "材料": ["材料1", "材料2"],
  "手順": ["手順1", "手順2"]
}
'''
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "醤油ラーメン"},
    ],
)
print(response.choices[0].message.content)

# 出力例
# ```json
# {
#   "材料": ["ラーメン", "鶏ガラスープ", "醤油", "ネギ", "メンマ", "チャーシュー", "海苔", "煮卵"],
#   "手順": ["鶏ガラスープを鍋に入れ、中火で温める。", "スープが温まったら、醤油を加えて味を調整する。", "別の鍋でラーメンを茹で、指定の時間が経ったらざるにあける。", "器に茹でたラーメンを盛り、スープを注ぐ。", "トッピングとしてネギ、メンマ、チャーシュー、海苔、煮卵をのせる。", "お好みで胡椒やごまを振りかけて完成。"]
# }
# ```
