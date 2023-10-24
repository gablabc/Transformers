from torch import nn
from utils import get_mnist_dataloaders
from models import Embeddings, PositionalEncoding, attention, subsequent_mask, MultiHeadedAttention
import matplotlib.pyplot as plt

# Call this once first to download the datasets
_ = get_mnist_dataloaders()

def plot_samples():
    a, _, _ , _, _, _= get_mnist_dataloaders()
    num_row = 2
    num_col = 5
    num_images = num_row * num_col
    _, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i, (x, y) in enumerate(a):
        if i >= num_images:
            break
        print(x.shape)
        ax = axes[i//num_col, i%num_col]
        x = (x.numpy().squeeze().reshape(28, 28) * 255).astype(int)
        y = y.numpy()[0]
        ax.imshow(x, cmap='gray')
        ax.set_title(f"Label: {y}")
        
    plt.tight_layout()
    plt.show()
    
plot_samples()


# Debug input encoding
one_batch_train, _, _, _, _, _ = get_mnist_dataloaders(batch_size=32)
input, target = next(iter(one_batch_train))
encoder = nn.Sequential(Embeddings(16, 3), PositionalEncoding(16, 0))
h = encoder(input)
h.shape

# Debug Attention Mechanisms
atte = attention(h, h, h, mask=subsequent_mask(784)) # self attention
print(atte[0].shape)  # transformed values
print(atte[1].shape)

# Debug Multi-Headed Attention
mh_module = MultiHeadedAttention(4, 16)
multi_atten = mh_module(h, h, h, mask=subsequent_mask(784))
print(mh_module)
print(multi_atten.shape)
print(mh_module.attn.shape)