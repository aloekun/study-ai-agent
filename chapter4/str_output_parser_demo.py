from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

ai_message = AIMessage(content="こんにちは。私はAIアシスタントです。")
output = output_parser.invoke(ai_message)
print(type(output))
print(output)

# 出力例
# <class 'str'>
# こんにちは。私はAIアシスタントです。
