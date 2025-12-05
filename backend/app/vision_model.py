from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

class VisionToPrompt:
    def __init__(self, model_dir="vision-model"):
        self.processor = LlavaProcessor.from_pretrained(model_dir)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float16)
        self.model.eval()
        if torch.cuda.is_available(): self.model.to("cuda")
    def generate_prompt(self, image: Image.Image, instruction="Describe this image in detail."):
        inputs = self.processor(images=image, text=instruction, return_tensors="pt")
        if torch.cuda.is_available(): inputs={k:v.to("cuda") for k,v in inputs.items()}
        with torch.no_grad(): gen=self.model.generate(**inputs, max_new_tokens=256)
        return self.processor.decode(gen[0], skip_special_tokens=True)
