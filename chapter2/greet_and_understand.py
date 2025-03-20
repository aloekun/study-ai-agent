from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "こんにちは！私はアロエといいます！"},
        {"role": "assistant", "content": "こんにちは、アロエさん！お会いできて嬉しいです。今日はどんなことをお話ししましょうか？"},
        {"role": "user", "content": "私のことがわかりますか。わかるなら名前の後ろに「君」を付けて読んでください。"}
    ],
)
print(response.to_json(indent=2))

# 出力例
# {
#   "id": "chatcmpl-BD8WBZk3pR71Dij36aXuefJroYuqH",
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "logprobs": null,
#       "message": {
#         "content": "もちろんです、アロエ君！何か特別なことをお話ししたいことはありますか？",
#         "refusal": null,
#         "role": "assistant",
#         "annotations": []
#       }
#     }
#   ],
#   "created": 1742470707,
#   "model": "gpt-4o-mini-2024-07-18",
#   "object": "chat.completion",
#   "service_tier": "default",
#   "system_fingerprint": "fp_3267753c5d",
#   "usage": {
#     "completion_tokens": 26,
#     "prompt_tokens": 92,
#     "total_tokens": 118,
#     "completion_tokens_details": {
#       "accepted_prediction_tokens": 0,
#       "audio_tokens": 0,
#       "reasoning_tokens": 0,
#       "rejected_prediction_tokens": 0
#     },
#     "prompt_tokens_details": {
#       "audio_tokens": 0,
#       "cached_tokens": 0
#     }
#   }
# }
