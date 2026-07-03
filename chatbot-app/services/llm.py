import requests
import time
import json
from config import Config

LLM_AVAILABLE = False
try:
    response = requests.get(f"{Config.API_BASE_URL}models", headers={"Authorization": f"Bearer {Config.API_KEY}"}, timeout=5)
    if response.status_code == 200:
        LLM_AVAILABLE = True
        print("✓ LLM API is accessible!")
    else:
        print("✗ LLM API returned error:", response.status_code)
except Exception as e:
    print(f"✗ Could not connect to LLM API: {e}")


def make_api_request_with_retry(payload, headers, max_retries=3, initial_wait=1):
    """Make API request with automatic retry on failure."""
    is_stream = payload.get('stream', False)
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{Config.API_BASE_URL}chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
                stream=is_stream
            )
            
            if response.status_code == 200:
                return response
            
            # If not 200, try to get error message
            error_msg = f"API Error {response.status_code}"
            try:
                error_msg += f": {response.text[:200]}"
            except:
                pass
                
            print(f"⚠️ {error_msg} (attempt {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                time.sleep(wait_time)
                continue
            
            return None
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                print(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None
    
    return None


def get_llm_response(prompt: str, system_message: str = None, history: list = None):
    """Get response from LLM API using direct HTTP requests (Streaming)."""
    if not LLM_AVAILABLE:
        yield "I'm unable to access the LLM service at the moment. Please try booking an appointment with our doctors."
        return
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    if history:
        # Context summarization (Phase 4 fix)
        mapped_history = [{"role": "assistant" if msg["role"] == "bot" else msg["role"], "content": msg["content"]} for msg in history]
        if len(mapped_history) > 6:
            # Take last 6 verbatim
            recent = mapped_history[-6:]
            older_count = len(mapped_history) - 6
            messages.append({
                "role": "system",
                "content": f"[Earlier context: patient discussed {older_count} earlier messages before the current topic.]"
            })
            messages.extend(recent)
        else:
            messages.extend(mapped_history)
    
    messages.append({"role": "user", "content": prompt})
    
    headers = {
        "Authorization": f"Bearer {Config.API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": Config.MODEL,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
        "max_tokens": 800,
    }
    
    response = make_api_request_with_retry(payload, headers)
    
    if not response:
        yield "Error: Could not connect to AI service. Please try again."
        return

    # Parse streaming response
    try:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error parsing stream: {str(e)}")
        yield f" Error processing response: {str(e)}"

def resolve_llm_response(report):
    """Ensure LLM response is a plain string for JSON serialization."""
    if isinstance(report, str):
        return report
    try:
        return "".join(chunk for chunk in report)
    except TypeError:
        return str(report)
