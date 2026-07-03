import sys
sys.path.append('.')
from services.llm import get_llm_response

history = [{"role": "user", "content": "hi"}, {"role": "bot", "content": "hello"}]
try:
    for chunk in get_llm_response("how are you?", history=history):
        print(chunk)
except Exception as e:
    print("Exception", e)
