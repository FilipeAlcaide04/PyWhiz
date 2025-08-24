import requests
import json
import time
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field


# -------------------------------
# AIResponse dataclass
# -------------------------------
@dataclass
class AIResponse:
    """
    A dataclass to store the response from the AI model,
    including prompt, response, model details, and performance metrics.
    """
    prompt: str
    response: str
    model: str
    # The history of the conversation at the time this response was generated.
    history: List[Dict[str, str]] = field(default_factory=list)
    tokens_used: Optional[int] = None
    latency: Optional[float] = None
    error: Optional[str] = None


# -------------------------------
# One-off call to Ollama
# -------------------------------
def call_ollama(
    prompt: str,
    api_url: str,
    model: str,
    max_tokens: int = 512,
    context: str = "",
    stream: bool = True,
    timeout: int = 30,
    **kwargs: Any
) -> str:
    """
    Makes a single, non-session-based call to the Ollama API and returns the response text.

    This function is useful for simple, one-off queries where conversation history
    is not required.

    Args:
        prompt: The input prompt for the model.
        api_url: The endpoint for the Ollama API (e.g., http://localhost:11434/api/generate).
        model: The name of the model to use.
        max_tokens: The maximum number of tokens to generate in the response.
        context: Optional additional context to provide to the model.
        stream: If True, streams the response incrementally.
        timeout: The request timeout in seconds.
        **kwargs: Additional parameters to pass to the Ollama API.

    Returns:
        The generated response text from the model, or an error message.
    """
    try:
        # Combine context and prompt if context is provided
        full_prompt = f"{context}\n{prompt}".strip() if context else prompt
        payload = {"model": model, "prompt": full_prompt, "stream": stream, **kwargs}

        # The 'max_tokens' parameter is not standard in Ollama's /api/generate
        # It's usually controlled by 'num_predict' in the 'options' dictionary.
        if 'options' not in payload:
            payload['options'] = {}
        payload['options']['num_predict'] = max_tokens

        with requests.post(api_url, json=payload, stream=stream, timeout=timeout) as response:
            response.raise_for_status()

            answer_parts: List[str] = []
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        answer_parts.append(data["response"])
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    # Ignore lines that are not valid JSON
                    continue

            return "".join(answer_parts).strip()

    except requests.exceptions.Timeout:
        return "Error: Timeout during the call to the Ollama API."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the Ollama API. Is the server running?"
    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama API: {e}"


# ---------------------------------------
# Session-based conversation
# ---------------------------------------
class OllamaSession:
    """
    Manages a conversational session with the Ollama API, maintaining history.
    """
    def __init__(
        self,
        model: str,
        api_url: str = "http://localhost:11434/api/generate",
        keep_last_n: Optional[int] = None,
    ):
        """
        Initializes the conversation session.

        Args:
            model: The name of the model to use for the session.
            api_url: The endpoint for the Ollama API.
            keep_last_n: The number of recent conversation turns to keep in the context.
                         If None, the entire history is used.
        """
        self.model = model
        self.api_url = api_url
        self.history: List[AIResponse] = []
        self.keep_last_n = keep_last_n

    def ask(
        self,
        prompt: str,
        context: str = "",
        max_tokens: int = 512,
        timeout: int = 30,
        **kwargs: Any
    ) -> AIResponse:
        """
        Sends a prompt to Ollama, maintains conversation history, and returns an AIResponse object.

        Args:
            prompt: The user's prompt.
            context: Optional system-level or initial context for the conversation.
            max_tokens: The maximum number of tokens for the response.
            timeout: The request timeout in seconds.
            **kwargs: Additional parameters for the Ollama API.

        Returns:
            An AIResponse object containing the full details of the interaction.
        """
        # Determine the relevant history for context
        history_for_context = self.history[-self.keep_last_n:] if self.keep_last_n is not None else self.history

        # Create a simple string-based history for the API context
        combined_history = "\n".join([f"User: {h.prompt}\nAssistant: {h.response}" for h in history_for_context])
        full_context = f"{context}\n{combined_history}".strip() if combined_history else context

        # --- FIX: Capture the history *before* the new call ---
        # Create a serializable version of the history for the AIResponse object
        history_list_of_dicts = [{"prompt": h.prompt, "response": h.response} for h in self.history]

        # Measure latency of the API call
        start_time = time.time()
        response_text = call_ollama(
            prompt,
            api_url=self.api_url,
            model=self.model,
            context=full_context,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )
        latency = time.time() - start_time

        # Create the response object, now including the history
        ai_response = AIResponse(
            prompt=prompt,
            response=response_text,
            model=self.model,
            latency=latency,
            history=history_list_of_dicts # Pass the captured history here
        )

        # Append the new interaction to the session's history
        self.history.append(ai_response)
        return ai_response

    def get_history(self) -> List[AIResponse]:
        """Returns the full session history."""
        return self.history

    def clear_history(self):
        """Clears the session's conversation history."""
        self.history = []
        print("Session history has been cleared.")



