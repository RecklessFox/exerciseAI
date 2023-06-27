import sys
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from diffusers import DiffusionPipeline

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(model_path, sequence, max_length):
    
    task = "You are a song lyrics generator. You take 2 verses of a song as input and you will generate the rest of the song in the genre of Taylor Swift. The 2 verses are: "
    prompt = sequence
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{prompt}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

def generate_image(sequence):
    
    task = "You are a Taylor Swift album cover for the album: "
    pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker = None
    )
    pipe.to("cuda")
    prompt = task + sequence
    image = pipe(prompt).images[0]
    image.save("cover.jpg")

if __name__ == "__main__":
    main(sys.argv[1:])

