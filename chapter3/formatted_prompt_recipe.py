from openai import OpenAI

client = OpenAI()

prompt = '''\
以下の料理のレシピを考えてください。

料理名： """
{dish}
"""
'''


# 料理名を指定してレシピを生成する関数
# dish: 料理名
# return: レシピ
# ユーザーの入力を使って、事前に作ったプロンプトと組み合わせてエージェントからの出力を得る
def generate_recipe(dish: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt.format(dish=dish)},
        ],
    )
    return response.choices[0].message.content


recipe = generate_recipe("シチュー")
print(recipe)

# 出力例
# シチューは温かくて心を満たしてくれる料理ですね。ここでは、基本的なビーフシチューのレシピを紹介します。野菜やお肉の旨味が凝縮された、家庭で作ることができるシチューです。

# ### ビーフシチューのレシピ

# #### 材料（4人分）
# - 牛肉（シチュー用）: 500g
# - 玉ねぎ: 1個
# - にんじん: 2本
# - じゃがいも: 2個
# - セロリ: 1本
# - マッシュルーム: 200g（お好みで）
# - にんにく: 2片
# - 小麦粉: 大さじ2
# - 赤ワイン: 200ml
# - ビーフブロス（もしくは水）: 500ml
# - トマトペースト: 大さじ2
# - ローリエ: 1枚
# - 塩: 適量
# - 黒こしょう: 適量
# - オリーブオイル: 大さじ2
# - バター: 大さじ1
# - パセリ（飾り用）: 適量

# #### 作り方

# 1. **下ごしらえ**
#    - 牛肉は一口大に切り、塩と黒こしょうをふる。
#    - 玉ねぎ、にんじん、じゃがいも、セロリはそれぞれ1cm角に切る。にんにくはみじん切りにする。
#    - マッシュルームは薄切りにする。

# 2. **牛肉を焼く**
#    - 大きめの鍋にオリーブオイルを熱し、牛肉を入れて表面がこんがりと焼けるまで焼く（約5分）。焼き色がついたら、一旦取り出す。

# 3. **野菜を炒める**
#    - 同じ鍋にバターを追加し、玉ねぎ、にんにくを炒める。玉ねぎが透明になるまで炒めたら、にんじん、セロリ、マッシュルームを加えてさらに炒める。

# 4. **小麦粉を加える**
#    - 野菜がしんなりしてきたら、小麦粉を加え、全体に絡めて軽く炒める。

# 5. **煮込む**
#    - 焼いた牛肉を鍋に戻し、赤ワインを加えて一煮立ちさせる。アルコールを飛ばした後、ビーフブロス、トマトペースト、ローリエを加え、弱火で蓋をして約1時間煮込む。

# 6. **入れた野菜を追加**
#    - じゃがいもを追加して、さらに30分程度煮込む。途中で味見をし、必要に応じて塩と黒こしょうで調整する。

# 7. **仕上げ**
#    - 煮込みが完了したら、ローリエを取り出し、器に盛り付ける。お好みで刻んだパセリを散らして完成！

# #### 提供方法
# ビーフシチューは、パンやライスとともに楽しむことができます。また、冷蔵庫で保存しておくと、味がさらに深まりますので、作り置きにもぴったりです。

# おいしいシチューをお楽しみください！
