import torch
import time
from model_zoo.language_model import LanguageModel
from vllm import LLM, SamplingParams


class VllmModel(LanguageModel):
    def __init__(
        self,
        model,
        temperature=0,
        num_gpus=2,
        top_k=-1,
        top_p=1,
        seed=42,
        sampling_params=None,
        revision=None,
        **kwargs,
    ):
        """
        Initialize the VllmModel with a Hugging Face model.

        Parameters:
            model (str): The model identifier or path to the pretrained model.
            tokenizer (AutoTokenizer, optional): If provided, use this tokenizer; otherwise, load one automatically.
            **kwargs: Additional configuration parameters (e.g., torch_dtype, quantization_config, etc.)
        """
        super().__init__(model)
        self.sampling_params = sampling_params
        self.model = LLM(model, tensor_parallel_size=num_gpus, revision=revision)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        vars(self).update(kwargs)
        if temperature is not None:
            if sampling_params is not None:
                self.sampling_params.temperature = temperature
            else:
                self.sampling_params = SamplingParams(temperature=temperature)

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def __str__(self):
        return f"VllmModel(model={self.model.__str__()})"

    def __repr__(self):
        return str(self)

    def config_model(self, **kwargs):
        generation_config_keys = {"n", "top_k", "top_p", "temperature"}

        self.sampling_params = SamplingParams(
            **{k: v for k, v in kwargs.items() if k in generation_config_keys}
        )

        return self  # Allow method chaining

    def run(
        self,
        eval_examples,
        system_prompt=None,
        temperature=0,
        top_p=1,
        top_k=-1,
        max_tokens=1024,
        n=1,
        **kwargs,
    ):
        """
        Run the model on a list of evaluation examples.

        Parameters:
            top_k: int, optional: The number of highest probability vocabulary tokens to keep for top-k sampling.
            eval_examples (list): A list of dictionaries, each with keys "input" and "output".
            system_prompt (str): A system prompt to include in the messages (optional).
            temperature (float): Sampling temperature.
            top_p (float): Top-p nucleus sampling value.
            max_tokens (int): Maximum new tokens to generate.
            n (int): Number of samples to generate.

        Returns:
            A tuple (outputs, answers, reasoning_output) where:
              - outputs: The generated outputs from the model.
              - answers: The expected answers from the evaluation examples.
              - reasoning_output: This is always empty for vllm models.
        """
        all_outputs = []
        answers = []
        messages = []
        tokens = {"input_token": [], "output_token": []}
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature or self.temperature or 0,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if self.sampling_params:
            if (
                temperature == 0
                and top_p == 1
                and top_k == -1
                and max_tokens == 1024
                and n == 1
            ):  # default
                sampling_params = self.sampling_params
        for example in eval_examples:
            if system_prompt:
                messages.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": example["input"]},
                    ]
                )
            else:
                messages.append([{"role": "user", "content": example["input"]}])
            answers.append(example["output"])

        resps = self.model.chat(messages=messages, sampling_params=sampling_params)

        outputs = [[out.text for out in response.outputs] for response in resps]
        all_outputs.extend(outputs)

        # Count tokens for both input and output
        for example, outs in zip(eval_examples, outputs):
            # Combine system prompt with user input if available
            input_text = f"{system_prompt} {example['input']}" if system_prompt else example["input"]
            input_token_count = len(input_text.split())
            for out in outs:
                output_token_count = len(out.split())
                tokens["input_token"].append(input_token_count)
                tokens["output_token"].append(output_token_count)

        if not all(all_outputs):
            print("empty response detected")
        latencies = [response.metrics.last_token_time-response.metrics.first_token_time for response in resps]
        return all_outputs, answers, latencies, tokens

    def query_once(
        self,
        messages,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        **kwargs,
    ):
        """
        A single query using the Hugging Face model's generate method.

        Parameters:
            top_k: int, optional: The number of highest probability vocabulary tokens to keep for top-k sampling.
            messages (list): A list of message dictionaries representing the conversation.
                             Example: [{"role": "user", "content": "Hello, how are you?"}]
            temperature (float, optional): Sampling temperature. Defaults to self.model.generation_config.temperature.
            top_p (float, optional): Top-p nucleus sampling parameter. Defaults to 1.0.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 256.
            **kwargs: Additional keyword arguments to pass to the generate method.

        Returns:
            str or None: The decoded response from the model, or None if an error occurs.
        """
        self.sampling_params.update_from_generation_config(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
            }
        )
        try:
            # Use the tokenizer to apply the chat template and prepare input tensors.
            # Here we wrap messages in a list to create a batch of one query.
            # Generate output tokens using the model.
            output = self.model.chat(messages, sampling_params=self.sampling_params)
            # Skip the prompt tokens and decode only the newly generated tokens.
            return output
        except Exception as e:
            print(f"⚠️ query_once encountered an error: {e}")
            return None
