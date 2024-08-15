import logging
import os
from typing import Any
os.environ["OPENAI_API_KEY"] = "sk-xe4yY6HmC-AIbUkRK27iUPzhisihhTzqap2QU4ue84T3BlbkFJidCVFBINiiC9nOTiXqeQMz1MHgw_O5pjElVIPqwK4A"
from openai import OpenAI, OpenAIError
from evals.api import CompletionFn, CompletionResult
from evals.record import record_sampling
import json


class OpenAIChatCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                if choice.message.content is not None:
                    completions.append(choice.message.content)
        return completions

class GPT35TurboCompletionFn(CompletionFn):
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        **kwargs,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def __call__(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs,
    ) -> OpenAIChatCompletionResult:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return OpenAIChatCompletionResult(raw_data={}, prompt=str(prompt))

        if not response:
            logging.error(f"No valid completions returned for prompt: {prompt}")
            return OpenAIChatCompletionResult(raw_data={}, prompt=str(prompt))

        result = OpenAIChatCompletionResult(raw_data=response, prompt=str(prompt))
        completions = result.get_completions()

        if not completions:
            logging.error(f"No completions returned for prompt: {prompt}")
            return OpenAIChatCompletionResult(raw_data={}, prompt=str(prompt))

        record_sampling(
            prompt=result.prompt,
            sampled=completions,
            model=getattr(result.raw_data, "model", "unknown_model"),
            usage=getattr(result.raw_data, "usage", {})
        )

        return result