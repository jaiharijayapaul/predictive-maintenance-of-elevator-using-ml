import os
from groq import Groq

# Cached system prompt — built once, reused across all requests
_SYSTEM_PROMPT_CACHE: str | None = None

def _build_system_prompt() -> str:
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE

    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
    except Exception:
        readme_content = "Elevator Predictive Maintenance System details are currently unavailable."

    _SYSTEM_PROMPT_CACHE = f"""You are the official AI Assistant for the 'Elevator Predictive Maintenance System' dashboard.
Your role is to assist clients, testers, and developers with questions about this system.
You are an expert in machine learning, predictive maintenance, and the specific architecture of this project.

Here is the complete project documentation for your reference:
---
{readme_content}
---

Response Guidelines:
1. Answer project-specific questions using the documentation above (models, features, SMOTE, architecture, pages, etc.).
2. For general ML or elevator maintenance questions, draw from your broader expertise and connect it to this project.
3. Keep answers concise and well-structured using Markdown (headers, bullet points, bold text).
4. Do NOT repeat the question. Get straight to the answer.
5. If unsure, say so honestly."""
    return _SYSTEM_PROMPT_CACHE


class ElevatorChatbot:
    # Models in preference order — Llama 3.3 70B is highly capable
    MODEL_CANDIDATES = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.system_prompt = _build_system_prompt()
        self.model = self._find_working_model()
        self.history: list[dict] = []  # maintained in-memory per session

    def _find_working_model(self) -> str:
        """Probe each model candidate and return first working one."""
        for model_name in self.MODEL_CANDIDATES:
            try:
                self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5,
                )
                return model_name
            except Exception:
                continue
        raise RuntimeError(
            "No working Groq model found. "
            "Please check your API key at https://console.groq.com/keys"
        )

    def create_chat_session(self):
        """Reset and return a fresh chat history (list of messages)."""
        self.history = []
        return self.history

    def send_message_stream(self, chat_session, message: str):
        """Append user message, call Groq with streaming, yield text chunks."""
        chat_session.append({"role": "user", "content": message})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *chat_session,
            ],
            stream=True,
            temperature=0.7,
            max_tokens=1024,
        )

        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            yield delta

        # Save assistant reply into history for multi-turn context
        chat_session.append({"role": "assistant", "content": full_response})
