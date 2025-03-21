from openai import OpenAI

client = OpenAI()

# 細かい条件を指定せず、ステップバイステップで進めることだけを指示するサンプル
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "ステップバイステップで考えてください。"},
        {"role": "user", "content": "3 + 4 * 5 - 2 * 6"},
    ],
)
print(response.choices[0].message.content)

# 出力例
# この数学の計算問題を解くために、演算の順序に従います。通常、先に乗算と除算を行い、次に加算と減算を行います。

# 1. まず、乗算を計算します。
#    - \(4 * 5 = 20\)
#    - \(2 * 6 = 12\)

# 2. それを式に代入します。
#    - \(3 + 20 - 12\)

# 3. 次に、加算と減算を左から右へ行います。
#    - \(3 + 20 = 23\)
#    - \(23 - 12 = 11\)

# したがって、最終的な答えは **11** です。
