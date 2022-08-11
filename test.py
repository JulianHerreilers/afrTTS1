import torch
import soundfile as sf
from univoc import Vocoder
from tacotron import load_afrdict, text_to_id, Tacotron
import matplotlib.pyplot as plt
from IPython.display import Audio
vocoder = Vocoder.from_pretrained("https://github.com/bshall/UniversalVocoding/releases/download/v0.2/univoc-ljspeech-7mtpaq.pt").cuda()
print("1")
tacotron = Tacotron.from_pretrained(
    "https://github.com/bshall/Tacotron/releases/download/v0.1/tacotron-ljspeech-yspjx3.pt"
).cuda()
print("2")
cmudict = load_afrdict()
cmudict["PYTORCH"] = "P AY1 T AO2 R CH"
text = "Testing Pytorch by location."
print("3")

x = torch.LongTensor(text_to_id(text, cmudict)).unsqueeze(0).cuda()
with torch.no_grad():
    mel, alpha = tacotron.generate(x)
    wav, sr = vocoder.generate(mel.transpose(1, 2))
print("4")


Audio(wav, rate=sr)

plt.imshow(alpha.squeeze().cpu().numpy(), vmin=0, vmax=0.8, origin="lower")
plt.xlabel("Decoder steps")
plt.ylabel("Encoder steps")
