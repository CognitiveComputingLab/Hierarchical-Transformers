'''
This file contains a trainable implementation of the full architecture, as described in Subsection 3.1 (Foundational Methods).
The hyperparameters are set as they were for the experiments discussed in the paper.
The structure of this file is as follows:

DATALOADER --------------------------------------------------
- CLASS: CIFAR10DataModule

ENCODER-DECODER ---------------------------------------------
- CLASS: TransformerBlock (used in both the encoder and the decoder)
- FUNCTION: _gen_timing_signal (taken from https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/common_layer.py)

ENCODER-ONLY -------------------------------------------------
- CLASS: PositionalImageEmbedding
- FUNCTION: select_token_pairs_for_merging
- CLASS: MergingBlock
- CLASS: EncoderArchitecture

DECODER-ONLY -------------------------------------------------
- CLASS: DecompositionBlock (aka the unmerging block)
- CLASS: DecoderArchitecture

TRAINING-----------------------------------------------------
- CLASS: Model
- FUNCTION: train
- main
-------------------------------------------------------------

'''


############# DATALOADER #############

%pip install -q pytorch_lightning wandb einops

import os

import math
from math import sqrt

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision
from torchvision import transforms

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import LambdaLR

import wandb

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizing in range [-1, 1] for all 3 channels
        ])

    def prepare_data(self):
        if not os.path.exists('./data'):
            os.makedirs('./data')

        torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            original_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = torch.utils.data.random_split(original_train, [45000, 5000])

        if stage == 'test' or stage is None:
            self.cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=4)


############# ENCODER-DECODER #############

class TransformerBlock(nn.Module):
    ''' Transformer encoder block.'''
    def __init__(self, n_embed, num_heads, dropout=0.0):
        '''
        Pre-norm formulation.
        Feed-forward hidden layer is 4x n_embed.
        '''

        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.self_attention = nn.MultiheadAttention(n_embed, num_heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.layer_norm_1(x)
        attention_out, attention_weights = self.self_attention(norm_x, norm_x, norm_x)
        x = x + attention_out
        x = x + self.mlp(self.layer_norm_2(x))
        return x


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    '''
    Generates a [1, length, channels] timing signal consisting of sinusoids.
    Taken from:
    https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/common_layer.py
    Which in turn was adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    '''

    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


############# ENCODER #############

def select_token_pairs_for_merging(logits, r):
    '''
    Given logits and some r <= T // 2, this function uses a Softmax obtain a policy that is then sampled from. 

    We retain these sampled indices to allow us to backprop through our log-probs.

    In this function, no gradient tracking is required since we soley use the
    tensors obtained to re-organise (& backprop through) tensors that do
    require gradients.

    We can think of this function as performing sampling from a constrained distribution where the logits influence 
    the probability of being sampled. 
    '''

    batch_size, num_tokens, _ = logits.size()

    # We perform all operations on a cloned and detached version of the logits.
    # Gradients are obtained from the log-probs (outside of this function).
    masked_logits = logits.clone().detach().to(device='cuda')
    # We now mask the diagonal here since this was interferring with learning.
    masked_logits.diagonal(dim1=-2, dim2=-1).fill_(-float('inf'))

    with torch.no_grad():
        # Initialize a mask to keep track of selected tokens
        mask = torch.zeros_like(logits, dtype=torch.bool, device='cuda')

        # Tensor to store the pairs of tokens selected for merging
        selected_pairs = torch.zeros(batch_size, r, 2, dtype=torch.long, device='cuda')

        # Tensor to track the indices selected in each batch
        indices_batch = torch.zeros(batch_size, r, dtype=torch.int64, device='cuda')

        for pair_idx in range(r):
            # Set already selected tokens' similarities to -inf (becuase they then go into a softmax)
            masked_logits.masked_fill_(mask, float('-inf'))

            # Re-normalise using a Softmax
            policy = torch.nn.functional.softmax(masked_logits.view(batch_size, -1), dim=-1)
            indices = torch.multinomial(policy, 1).squeeze(-1)

            # Tracking the indices
            indices_batch[:, pair_idx] = indices

            # No need to worry about gradients here, since "indices" was initialised in a torch.no_grad()
            rows = torch.div(indices, num_tokens, rounding_mode='trunc')
            cols = indices % num_tokens

            # Store the selected token pairs
            selected_pairs[:, pair_idx, 0] = rows
            selected_pairs[:, pair_idx, 1] = cols

            # Update the mask to avoid selecting these tokens again
            mask[torch.arange(batch_size), rows, :] = True
            mask[torch.arange(batch_size), :, cols] = True
            mask[torch.arange(batch_size), cols, :] = True
            mask[torch.arange(batch_size), :, rows] = True

    return selected_pairs, indices_batch


class PositionalImageEmbedding(nn.Module):
    ''' Performs the patching / tokenization. '''
    def __init__(self, n_embed, image_size=(32,32), patch_size=(2,2), channels=3, bands=8):
        super().__init__()
        self.ff = self.fourier_features(image_size, bands)

        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = (channels + 4*bands + 2) * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, n_embed),
            nn.LayerNorm(n_embed),
        )

        # Generate and register position encoding as a buffer
        position_encoding = self.fourier_features(image_size, bands)
        self.register_buffer("position_encoding", position_encoding)

    def fourier_features(self, shape, bands):
        height, width = shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')

        # Linearly spaced frequencies
        # Minimum frequency for a full oscillation over the dimension
        min_freq = 1. / max(height, width)
        # Nyquist frequency
        max_freq = min(height, width) / 2
        freqs = torch.linspace(min_freq, max_freq, steps=bands)

        freqs_y = freqs.view(-1, 1, 1).repeat(1, height, 1)
        freqs_x = freqs.view(-1, 1, 1).repeat(1, width, 1)

        embeddings = torch.cat([
            torch.sin(2 * math.pi * y.unsqueeze(0) * freqs_y),
            torch.cos(2 * math.pi * y.unsqueeze(0) * freqs_y),
            torch.sin(2 * math.pi * x.unsqueeze(0) * freqs_x),
            torch.cos(2 * math.pi * x.unsqueeze(0) * freqs_x),
            y.unsqueeze(0),  # Add y coordinates
            x.unsqueeze(0)   # Add x coordinates
        ], dim=0)
        return embeddings

    def forward(self, img):
        # Initial x of shape [batch_size x channels x height x width]
        # Create position encoding of the same shape as x and move to the correct device
        batch_size = img.shape[0]
        enc = self.position_encoding.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Concatenate position encoding along the channel dimension
        # Shape is now [batch_size x (channels + 4*bands) x height x width]
        x = torch.cat([img, enc], dim=1)

        # Reshape into a sequence of patches
        x = self.to_patch_embedding(x)

        return x


class MergingBlock(nn.Module):
    ''' Merging block.'''

    def __init__(self, n_embed, num_heads, dropout=0.0):
        super().__init__()

        # Where these q's (q_intial) are used for voting
        self.qk_initial = nn.Linear(n_embed, 2*n_embed, bias=None)
        self.kv = nn.Linear(n_embed, 2*n_embed, bias=None)
        # Where these q's (q_cross_attention) are used for cross attention post merging
        self.q = nn.Linear(n_embed, n_embed, bias=None)

        # Learned token merging function
        self.merger = nn.Sequential(
            nn.Linear(2*n_embed, 4*n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

        # Output projection
        self.out = nn.Linear(n_embed, n_embed)

        # Layer norms
        self.layer_norm_q = nn.LayerNorm(n_embed)
        self.layer_norm_0 = nn.LayerNorm(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

        # Transformer MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # Hyperparams
        self.n_embed = n_embed
        self.num_heads = num_heads

    def forward(self, x, r):
        # Batch, num_tokens, n_embed
        B, T, C = x.size()

        assert r <= T // 2, 'r must be <= T/2'

        q_initial, k_initial = self.qk_initial(self.layer_norm_q(x.detach())).split(self.n_embed, dim=2)
        k, v = self.kv(self.layer_norm_0(x)).split(self.n_embed, dim=2)

        # -> (B, num_heads, num_tokens, head_dimension)
        q_initial = q_initial.view(B, T, 1, C).transpose(1, 2)
        k_initial = k_initial.view(B, T, 1, C).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Calculate attention scores
        scores = (q_initial @ k_initial.transpose(-2, -1)) * (1.0 / sqrt(k_initial.size(-1)))

        logits = scores.squeeze()

        # This is the matrix from which we obtain the log probabilities
        log_policy_flat = F.log_softmax(logits.view(B, -1), dim=1)

        # We now have logits and log_probs.
        # The logits will be used for sampling (via Softmax).
        # The log_probs will be backpropagated through using the indices obtained via sampling.

        # Obtain pairs to merge
        # Here we pass our log_probs tensor through the function to update it with this set of actions
        pairs_batch, indices = select_token_pairs_for_merging(logits, r)

        # Here we backprop through these decisions (later we manually re-scale the gradients by the full MSE loss)
        sampled_log_probs = torch.gather(log_policy_flat, 1, indices)
        merging_decision_loss = sampled_log_probs.sum()

        # Helpful visualisation
        #make_dot(merging_decision_loss, params=dict(list(self.named_parameters()))).render("computation_graph", format="png")

        # We don't want to try and backward when performing validation or inference
        if merging_decision_loss.requires_grad:
            merging_decision_loss.backward()

        # Concatenate tokens to be merged
        merged_mask = torch.ones(B, T, dtype=torch.bool, device='cuda')
        pairs = x.gather(1, pairs_batch.view(B, -1).unsqueeze(-1).expand(-1, -1, self.n_embed))
        pairs = pairs.view(B, r, 2 * self.n_embed)

        # Track which tokens are not being merged
        merged_mask.scatter_(1, pairs_batch.view(B, -1), False)

        # Merge tokens
        merged_tokens = self.merger(pairs)

        # Organise everything ready for output
        remaining_tokens = x[merged_mask].view(B, T - 2 * r, self.n_embed)
        # Our next layer of tokens (pre a final cross attention)
        x_prime = torch.cat([merged_tokens, remaining_tokens], dim=1)

        # Performing cross-attention on our new representations
        q_cross_attention = self.q(self.layer_norm_1(x_prime))
        q_cross_attention = q_cross_attention.view(B, T-r, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Calculate attention weights
        att = (q_cross_attention @ k.transpose(-2, -1)) * (1.0 / sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attention_dropout(att)
        y = att @ v # (B, num_heads, T, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T-r, C) # re-assembe head outputs side by side

        # Output projection for attention
        y = self.residual_dropout(self.out(y))

        # Residual connections for attention and MLP outputs
        x_prime = x_prime + y
        x_prime = x_prime + self.mlp(self.layer_norm_2(x_prime))

        return x_prime


class EncoderArchitecture(nn.Module):
    def __init__(self, *, image_size, patch_size, fourier_bands, n_embed, num_layers, num_heads, channels=3, dropout=0.0):
        super().__init__()

        assert n_embed % num_heads == 0, 'n_embed must be divisible by num_heads'

        self.timing_signal = _gen_timing_signal(num_layers, n_embed).to('cuda')

        self.to_patch_embedding = PositionalImageEmbedding(n_embed=n_embed, image_size=image_size, patch_size=patch_size, channels=channels, bands=fourier_bands)

        self.dropout = nn.Dropout(dropout)

        self.transformer_block = TransformerBlock(n_embed, num_heads, dropout)
        self.merging_block = MergingBlock(n_embed, num_heads, dropout)


    def forward(self, img, merging_schedule):
        x = self.to_patch_embedding(img)

        B, T, C = x.shape

        x = self.dropout(x)

        for l, r in enumerate(merging_schedule):
            # Signal to allow the network to differentiate between layers (broadcasts along batch dimension)
            x += self.timing_signal[:, l, :].unsqueeze(1).repeat(1, x.shape[1], 1)

            x = self.transformer_block(x)
            x = self.merging_block(x, r)

        return x


############# DECODER #############

class DecompositionBlock(nn.Module):
    def __init__(self, n_embed, num_heads, dropout=0.0):
        super().__init__()

        self.scoring_function = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embed, 1),
            nn.Dropout(dropout)
        )

        self.unmerger = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embed, 2*n_embed),
            nn.Dropout(dropout)
        )

        # --- Cross attention stuff ---

        self.kv = nn.Linear(n_embed, 2*n_embed, bias=None)
        self.q = nn.Linear(n_embed, n_embed, bias=None)
        self.out = nn.Linear(n_embed, n_embed)

        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

        self.layer_norm_0 = nn.LayerNorm(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.n_embed = n_embed
        self.num_heads = num_heads

    def forward(self, x, r):

        B, T, C = x.shape

        # Keys & values for cross attention
        k, v = self.kv(self.layer_norm_0(x)).split(self.n_embed, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, num_heads, num_tokens, head_dimension)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, num_heads, num_tokens, head_dimension)

        # --- Unmerging process ---

        # Scoring each token
        scores = self.scoring_function(x.detach())

        policy = F.softmax(scores, dim=-2).squeeze(-1)
        log_policy = F.log_softmax(scores, dim=-2).squeeze(-1)

        # Picking highest r to unmerge
        _, indices = torch.topk(policy, r, dim=1)

        # Accumulating gradients for the unmerging decision loss
        sampled_log_probs = torch.gather(log_policy, 1, indices)
        unmerging_decision_loss = sampled_log_probs.sum()

        if unmerging_decision_loss.requires_grad:
            unmerging_decision_loss.backward()

        # Mask to avoid duplication
        mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        mask.scatter_(1, indices, False)

        # Gathering tokens to unmerge
        indices = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        top_tokens = torch.gather(x, 1, indices)

        # Unmerging these tokens
        unmerged = self.unmerger(top_tokens)

        # Reshape unmerged tokens: [batch_size, r, 2*n_embed] -> [B, 2*r, n_embed]
        unmerged = unmerged.view(B, r*2, self.n_embed)

        # Mask out the original tokens that were unmerged
        remaining_tokens = x[mask].view(B, T - r, self.n_embed)

        # Combine the original and unmerged tokens
        x_prime = torch.cat([remaining_tokens, unmerged], dim=1)

        # --- Cross attention ---

        q = self.q(self.layer_norm_1(x_prime))
        q = q.view(B, T+r, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Calculate attention weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attention_dropout(att)
        y = att @ v # (B, num_heads, T+r, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T+r, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T+r, C) # re-assembe head outputs side by side

        # Output projection for attention
        y = self.residual_dropout(self.out(y))

        # Residual connections for attention and MLP outputs
        x_prime = x_prime + y
        x_prime = x_prime + self.mlp(self.layer_norm_2(x_prime))

        return x_prime


class DecoderArchitecture(nn.Module):
    def __init__(self, *, image_size, patch_size, fourier_bands, n_embed, num_layers, num_heads, channels=3, dropout=0.0):
        super().__init__()

        self.timing_signal = _gen_timing_signal(num_layers, n_embed).to('cuda')

        self.decomposition_block = DecompositionBlock(n_embed, num_heads, dropout=0.0)
        self.transformer_block = TransformerBlock(n_embed, num_heads, dropout)
        self.image_embedding = PositionalImageEmbedding(n_embed=n_embed, image_size=image_size, patch_size=patch_size, channels=channels, bands=fourier_bands)
        self.height_width = image_size[0]

        # Cross attention to query pixel values from latent space

        self.fourier_to_n_emb = nn.Linear(4*(fourier_bands)+2, n_embed)

        self.self_attention = nn.MultiheadAttention(n_embed, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 2*n_embed),
            nn.GELU(),
            nn.Linear(2*n_embed, 3),
        )

        self.layer_norm_0 = nn.LayerNorm(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed)

        # To get the rgb values between -1 and 1
        self.tanh = nn.Tanh()

    def forward(self, x, merging_schedule):
        B, T, C = x.shape

        for l, r in enumerate(reversed(merging_schedule)):
            # Signal to allow the network to differentiate between layers (broadcasts along batch dimension)
            x += self.timing_signal[:, l, :].unsqueeze(1).repeat(1, x.shape[1], 1)

            x = self.decomposition_block(x, r)
            x = self.transformer_block(x)

        # Now we query this expanded latent with our handcrafted queries
        position_encoding = self.image_embedding.position_encoding
        position_encoding = rearrange(position_encoding, 'c h w -> (h w) c')
        position_encoding = self.fourier_to_n_emb(position_encoding) # [h*w, c] -> [h*w, n_embed]
        position_encoding = repeat(position_encoding, 'hw c -> b hw c', b=B)

        norm_kv = self.layer_norm_0(x)
        norm_q = self.layer_norm_1(position_encoding)

        # Compute MHA at n_embed
        attention_out, attention_weights = self.self_attention(norm_q, norm_kv, norm_kv)

        # Reduce the dimentions to RGB predictions
        x = self.tanh(self.mlp(attention_out))

        # Reshape to grid
        x = rearrange(x, 'b (h w) c -> b h w c', h=self.height_width, w=self.height_width)

        return x


############# TRAINING #############

class Model(L.LightningModule):

    def __init__(self, model_kwargs, lr, complete_merging_schedule, rho_0, rho_step):
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.encoder = EncoderArchitecture(**model_kwargs)
        self.decoder = DecoderArchitecture(**model_kwargs)

        self.rho = rho_0
        self.current_merging_schedule = complete_merging_schedule

        self.save_hyperparameters('lr', 'complete_merging_schedule', 'rho_0', 'rho_step', 'model_kwargs')

    def forward(self, x):
        compressed_representation = self.encoder(x, self.current_merging_schedule)
        reconstruction = self.decoder(compressed_representation, self.current_merging_schedule)
        return reconstruction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        images, _ = batch

        compressed_representation = self.encoder(images, self.current_merging_schedule)
        reconstructed = self.decoder(compressed_representation, self.current_merging_schedule)

        # Reshaping the images to match what the loss expects
        images = rearrange(images, 'b c h w -> b h w c')
        loss = F.mse_loss(reconstructed, images, reduction='mean')

        self.log(f'{mode}_loss', loss)
        self.log(f'schedule length', len(self.current_merging_schedule))
        return loss

    def generate_reconstructions(self, batch, num_samples=10):
        images, _ = batch
        compressed_representation = self.encoder(images, self.current_merging_schedule)
        reconstructed = self.decoder(compressed_representation, self.current_merging_schedule)

        # Select random samples
        indices = torch.randperm(images.size(0))[:num_samples]
        return images[indices], reconstructed[indices]

    def log_reconstructions(self, batch, step_type='test'):
        original, reconstructed = self.generate_reconstructions(batch)

        # torchvision.utils.make_grid expects channels first
        reconstructed = rearrange(reconstructed, 'b h w c -> b c h w')

        # Convert tensors to grid of images
        original_grid = torchvision.utils.make_grid(original)
        reconstructed_grid = torchvision.utils.make_grid(reconstructed)
        # Log to wandb
        self.logger.experiment.log({
            f"{step_type}_original_images": wandb.Image(original_grid),
            f"{step_type}_reconstructed_images": wandb.Image(reconstructed_grid)
        })

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()
        loss = self._calculate_loss(batch, mode='train')
        self.manual_backward(loss)

        # We manually scale the gradients of the q_intial and scoring_function networks before we step
        loss_value = loss.detach()

        for param in self.encoder.merging_block.qk_initial.parameters():
            if param.grad is not None:
                param.grad *= loss_value

        for param in self.decoder.decomposition_block.scoring_function.parameters():
            if param.grad is not None:
                param.grad *= loss_value

        # Step the optimiser
        optim.step()

        # Zero grads after we step so that we can accumulate gradients in the forward pass
        optim.zero_grad()

        return loss

    def validation_step(self, batch, batch_idx):
        # Validation set always uses the full merging schedule
        self.current_merging_schedule = self.hparams.complete_merging_schedule
        loss = self._calculate_loss(batch, mode='val')

        if batch_idx == 0:  # Logging reconstruction for first batch
            self.log_reconstructions(batch, 'validation')

        return loss

    def test_step(self, batch, batch_idx):
        # Test set always uses the full merging schedule
        self.current_merging_schedule = self.hparams.complete_merging_schedule
        loss = self._calculate_loss(batch, mode='test')

    def adjust_merging_schedule(self):
        # Dynamically adjust the merging_schedule based on the current value of rho
        max_length = len(self.hparams.complete_merging_schedule)
        probabilities = [self.rho ** i for i in range(max_length)]
        schedule_length = 1 + sum(np.random.rand() < p for p in probabilities)
        self.current_merging_schedule = self.hparams.complete_merging_schedule[:schedule_length]

    def on_train_batch_start(self, batch, batch_idx):
        self.adjust_merging_schedule()

    def on_train_epoch_end(self):
        self.rho = min(self.rho + self.hparams.rho_step, 1)  # Cap rho at 1
        self.log(f'rho', self.rho)


def train():
    # Hyperparameters
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-4
    RHO_0 = 0.05                                         # Initial rho
    RHO_STEP = 0.01                                      # Per epoch
    COMPLETE_MERGING_SCHEDULE = [128, 64, 32, 16, 8, 4, 2, 1]
    MODEL_KWARGS = {
        'image_size': (32,32),
        'patch_size': (2, 2),
        'fourier_bands': 64,
        'num_layers': 8,
        'n_embed': 256,
        'num_heads': 8,
        'dropout': 0.1,
    }

    # Initialise data, model and logger

    data = CIFAR10DataModule(batch_size=BATCH_SIZE)
    model = Model(MODEL_KWARGS, LR, COMPLETE_MERGING_SCHEDULE, RHO_0, RHO_STEP)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer setup
    trainer = Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    # Start training
    trainer.fit(model, data)

    # Optionally: Test the model after training
    #trainer.test(model)

# Main

wandb.login()
wandb_logger = WandbLogger(project='Reconstruction_Personal')

train()

wandb.finish()
