from openai import OpenAI

client = OpenAI()


# レシピ名を入力すると、その料理のレシピを生成する関数
# 引数: dish (str) - 料理名
# 戻り値: レシピ
# ユーザーの入力を使って、事前に作ったプロンプトと組み合わせてエージェントからの出力を得る
def generate_recipe(dish: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "ユーザーが入力した料理のレシピを考えてください。"},
            {"role": "user", "content": f"{dish}"},
        ],
    )
    return response.choices[0].message.content


recipe = generate_recipe("ロールキャベツ")
print(recipe)

# 出力例
# 【ロールキャベツのレシピ】

# 材料:
# - キャベツの葉: 1個
# - お肉（挽き肉など）: 300g
# - 玉ねぎ: 1個
# - にんにく: 2片
# - ご飯: 1カップ
# - 卵: 1個
# - 塩コショウ: 適量
# - トマトソース: 400g

# 作り方:
# 1. キャベツの葉を外側から順にはがします。大きめの鍋にたっぷりの湯を沸かし、キャベツの葉を1枚ずつゆでて柔らかくします。水で冷やして水気を切ります。
# 2. 玉ねぎとにんにくをみじん切りにし、お肉と一緒にフライパンで炒めます。塩コショウで味を整えます。
# 3. 炒めた具材にご飯を加え、よく混ぜ合わせます。卵も加えてさらに混ぜます。
# 4. キャベツの葉に具材を適量のせ、くるくると巻いてロール状にします。すべてのキャベツの葉を使い切るまで繰り返します。
# 5. ファイヤープルーフのお鍋にトマトソースを底に敷き、ロールキャベツを並べます。
# 6. 余ったトマトソースをロールキャベツの上にかけ、蓋をして中火で20〜30分間煮ます。
# 7. 火を止め、お皿に盛り付けて完成です。お好みでパセリや粉チーズをかけても美味しいです。
