"""EXPERIMENT 1: Text Generation using GPT-2.

AIM:
    To generate meaningful text from a given prompt using the
    pre-trained GPT-2 language model.

ALGORITHM:
    1. Install Transformers and PyTorch.
    2. Load the GPT-2 tokenizer and pre-trained model.
    3. Convert the input prompt into token IDs.
    4. Generate new tokens using sampling parameters.
    5. Decode and display the generated text.

Sample Output (illustrative, text varies by run):
    Artificial intelligence will change education because it can help
    students learn at their own pace. ...

RESULT:
    Thus, text generation using the pre-trained GPT-2 model was
    implemented and executed successfully.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main() -> None:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    prompt = "Artificial intelligence will change education because"
    ids = tok.encode(prompt, return_tensors="pt")
    out = model.generate(
        ids,
        max_length=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
    )
    print("Generated text:")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
