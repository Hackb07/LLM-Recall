# EXPERIMENT 5: Build Chatbot using GPT-2 or OpenAI API

## AIM

To build a simple conversational chatbot using GPT-2 that accepts user messages and maintains conversation history.

## ALGORITHM

1. Load the GPT-2 tokenizer and language model.
2. Initialize conversation history with an assistant greeting.
3. Append each user message to the conversation history.
4. Generate a reply using the stored context.
5. Display the reply and continue until the user exits.

## CODE

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main() -> None:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    history = "AI: Hello! How can I help?\n"

    print("Chatbot ready. (Enter bye/quit/exit to stop.)")
    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit", "bye"]:
            break
        history += f"User: {msg}\nAI:"
        ids = tok.encode(history, return_tensors="pt")[:, -700:]
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(out[0][ids.shape[1]:],
                           skip_special_tokens=True).split("\n")[0]
        print("AI:", reply)
        history += reply + "\n"


if __name__ == "__main__":
    main()
```

## OUTPUT

```
Chatbot ready. (Enter bye/quit/exit to stop.)
You: Hello
AI: Hello! How can I help you?
You: What is Python?
AI: Python is a programming language.
You: bye
```

(Enter bye/quit/exit to stop.)

## RESULT

Thus, a GPT-2 chatbot was created to accept user messages and generate conversational replies.
