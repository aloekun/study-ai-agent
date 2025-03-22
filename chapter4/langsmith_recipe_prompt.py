from langsmith import Client

# LangSmith で公開されたプロンプトを使用する
client = Client()
prompt = client.pull_prompt("oshima/recipe")

prompt_value = prompt.invoke({"dish": "ラタトゥイユ"})
print(prompt_value)

# 出力例
# messages=[SystemMessage(content='ユーザーが入力した料理のレシピを考えてください。', additional_kwargs={}, response_metadata={}), HumanMessage(content='ラタトゥイユ', additional_kwargs={}, response_metadata={})]
