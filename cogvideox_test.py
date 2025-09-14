import torch, imageio
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

prompt = "A corgi surfing a wave at sunset, cinematic, 4k"
pipe = CogVideoXPipeline.from_pretrained(
    "zai-org/CogVideoX-2b", torch_dtype=torch.bfloat16
).to("cuda")

video = pipe(
    prompt=prompt,
    num_frames=49,               # tune for length
    guidance_scale=4.5,          # tune for quality
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "corgi.mp4", fps=12)
print("Wrote corgi.mp4")
