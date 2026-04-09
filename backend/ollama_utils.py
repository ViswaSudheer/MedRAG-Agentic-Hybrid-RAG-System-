import requests
import time
import json
OLLAMA_URL = "http://localhost:11434"

# def generate_llm(prompt, model: str = "llama3:latest"):
#     response = requests.post(
#         f"{OLLAMA_URL}/api/generate",
#         json={"model": model, "prompt": prompt, "stream": False}
#     )
#     response.raise_for_status()
#     return response.json()["response"]

def generate_llm(prompt):
    import requests
    import json

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3.2:3b",   # ✅ EXACT match from your list
        "prompt": str(prompt),      # ✅ force string
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=6000)

        print("🔍 Status Code:", response.status_code)

        # 👇 VERY IMPORTANT (see actual error)
        if response.status_code != 200:
            print("❌ Ollama Raw Error:", response.text)
            return "LLM generation failed (server error)."

        data = response.json()
        # print("🧾 Full Ollama Response:", data)
        resp = data.get("response", "")

        if not resp.strip():
            print("⚠️ Empty response from Ollama")
            return "I don't know from the provided context."

        return resp

    except Exception as e:
        print("❌ Ollama Exception:", e)
        return "LLM generation failed (exception)."
    except Exception as e:
        print("❌ Ollama generation failed:", e)
        return "LLM generation failed."
    

def generate_llm_stream(prompt):
    import requests

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                print(data.get("response", ""), end="", flush=True)

    except Exception as e:
        print("❌ Streaming failed:", e)
