from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "こんにちは！私はアロエといいます！"},
    ],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)

# 出力例
# こんにちは、アロエさん！お会いできてうれしいです。今日はどんなことをお話ししたいですか？
