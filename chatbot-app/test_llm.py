import sys
sys.path.append('.')
from services.llm import get_llm_response

for chunk in get_llm_response("Hello"):
    print(chunk, end='', flush=True)
print()
