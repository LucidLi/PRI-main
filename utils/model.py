from typing import Any
from openai import OpenAI
from tenacity import retry, wait_chain, wait_fixed
import os

class GPT:
    def __init__(self, model_name, api=None, temperature=0, seed=0):
        self.model_name = model_name

        client_kwargs = {"api_key": api}
        if os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.client = OpenAI(**client_kwargs)

        self.T = temperature
        self.seed=seed

        self.last_call_time = 0
        self.min_interval = 2.0
        

    def __call__(self, prompt,  n:int=1, debug=False, **kwargs: Any) -> Any:

        prompt = [{'role':'user', 'content':prompt}]
        if debug:
            return self.client.chat.completions.create(messages=prompt, n=n, model=self.model_name, temperature=self.T, seed=self.seed, **kwargs)
        else:
            return self.call_wrapper(messages=prompt, n=n, model=self.model_name, temperature=self.T, seed=self.seed,  **kwargs)


    

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                [wait_fixed(5) for i in range(2)] +
                [wait_fixed(10)])) # not use for debug

    def call_wrapper(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)
    
    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]
    

def load_model(model_name, api_idx, **kwargs):
    if "gpt" in model_name and "gpt2" not in model_name:
        return GPT(model_name, **kwargs)
    else:
        raise ValueError("model_name invalid")
