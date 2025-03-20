from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "こんにちは！私はアロエといいます！"},
    ],
)
print(response.to_json(indent=2))

# 出力例
# {
#   "id": "chatcmpl-BD8TiwYFQwq1MFjPdOhG7bWoKWwdI",
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "logprobs": null,
#       "message": {
#         "content": "こんにちは、アロエさん！お会いできて嬉しいです。今日はどんなことをお話ししましょうか？",
#         "refusal": null,
#         "role": "assistant",
#         "annotations": []
#       }
#     }
#   ],
#   "created": 1742470554,
#   "model": "gpt-4o-mini-2024-07-18",
#   "object": "chat.completion",
#   "service_tier": "default",
#   "system_fingerprint": "fp_3267753c5d",
#   "usage": {
#     "completion_tokens": 29,
#     "prompt_tokens": 27,
#     "total_tokens": 56,
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
