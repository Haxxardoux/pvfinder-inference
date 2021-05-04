import jax
import jax.numpy as jnp

from flax import linen as nn        
from flax import optim

from functools import partial
import typing 

# === Components of larger models ===
class ConvBNrelu(nn.Module):
    out_channels: int
    kernel_size: int
        
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size, 
            padding='SAME',
        )(x)
        
#         x = nn.BatchNorm()(x)
        return nn.relu(x)
        
class ConvTransposeBNrelu(nn.Module):
    out_channels: int
    kernel_size: int
        
    @nn.compact
    def __call__(self, x,):
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=self.kernel_size, 
            padding='SAME',
        )(x)
        
        x = nn.relu(x)
        
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size, 
            padding='SAME',
        )(x)
        
#         x = nn.BatchNorm()(x)
        return nn.relu(x)
    
    
# === Autoencoder models ===
class UNet(nn.Module):
    n: int
        
    @nn.compact
    def __call__(self, x):
        '''
        Example usage: UNet(n=16)
        '''
        conv1 = ConvBNrelu(out_channels=self.n, kernel_size=5)(x)
        conv2 = ConvBNrelu(out_channels=self.n, kernel_size=5)(conv1)
        conv3 = ConvBNrelu(out_channels=self.n, kernel_size=7)(conv2)
        conv4 = ConvBNrelu(out_channels=self.n, kernel_size=7)(conv3)

        up1 = ConvTransposeBNrelu(out_channels=self.n, kernel_size=7)(conv4) 
        up2 = ConvTransposeBNrelu(out_channels=self.n, kernel_size=7)(jnp.concatenate([up1, conv4], axis=-1))
        up3 = ConvTransposeBNrelu(out_channels=self.n, kernel_size=7)(jnp.concatenate([up2, conv3], axis=-1))
        up4 = ConvTransposeBNrelu(out_channels=self.n, kernel_size=7)(jnp.concatenate([up3, conv2], axis=-1))
        return ConvBNrelu(out_channels=1, kernel_size=5)(jnp.concatenate([up4, conv1], axis=-1))
    

# === Traditional models ===
class SimpleCNN6Layer(nn.Module):
#     kernel_sizes: typing.List
    n: int
        
    @nn.compact
    def __call__(self, x):
        '''
        Example usage: SimpleCNN6Layer(kernel_sizes=[25, 15, 15, 15, 5], n=16)
        '''
#         for i, k_size in enumerate(self.kernel_sizes):
#             x = nn.Conv(
#                 features=self.n,
#                 kernel_size=k_size, 
#                 padding='SAME',
#             )(x)
#             x = nn.relu(x) if i != len(self.kernel_sizes) else x # skip relu in favor of softmax if its the last layer
#         return nn.softplus(x)

        x = nn.Conv(
            features=self.n,
            kernel_size=25, 
            padding='SAME',
        )(x)
        x = nn.relu(x)
        
        x = nn.Conv(
            features=self.n,
            kernel_size=15, 
            padding='SAME',
        )(x)
        x = nn.relu(x)
        
        x = nn.Conv(
            features=self.n,
            kernel_size=15, 
            padding='SAME',
        )(x)
        x = nn.relu(x)
        
        x = nn.Conv(
            features=self.n,
            kernel_size=15, 
            padding='SAME',
        )(x)
        x = nn.relu(x)
        
        x = nn.Conv(
            features=self.n,
            kernel_size=5, 
            padding='SAME',
        )(x)
        x = nn.relu(x)
        return x
        