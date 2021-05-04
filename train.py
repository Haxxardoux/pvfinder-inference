import jax
import jax.numpy as jnp
from flax import optim
import torch
from tqdm import tqdm
import time

from models import UNet, SimpleCNN6Layer
from dataloader import KDE_to_PV_Dataset

# Initialize dataloader
train_dataset = KDE_to_PV_Dataset('/share/lazy/will/data/June30_2020_80k_1.h5', masking=False, load_xy=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16)

# Initialize model and optimizer
variables = UNet(n=16).init({'params':jax.random.PRNGKey(0)}, jax.numpy.ones((16, 4000, 1)))
# optimizer = optim.Adam(learning_rate=3e-4).create(parameters)
optimizer = optim.Adam(learning_rate=3e-4).create(variables['params'])
optimizer = jax.device_put(optimizer)

# define loss 
def symmetrical_loss(pred, truth):
    valid = (~jnp.isnan(truth)).astype(jnp.float32)

    # Compute r, only including non-nan values. r will probably be shorter than x and y.
    r = jnp.abs((pred*valid + 1e-5) / (truth*valid + 1e-5))

    # Compute -log(2r/(r² + 1))
    alpha = -jnp.log(2 * r / (r ** 2 + 1))

    # Sum up the alpha values, and divide by the length of x and y. Note this is not quite
    # a .mean(), since alpha can be a bit shorter than x and y due to masking.
    # nanmean does the same thing
    return jax.numpy.nanmean(alpha)
    
# define the training step
@jax.jit
def train_step(optimizer, kdes, pvs):
    # defined inside so we can @jax.jit the entire thing instead of the pieces individually. i think...
    def loss_fn(params):
        pred = UNet(n=16).apply({'params': params}, kdes) # i tried to not make this depend on the model (pass UNet(n=16) as argument) but it does not work since this gets compiled 
        return symmetrical_loss(pred, pvs), pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss

iterator = tqdm(train_loader)
for batch in iterator:
#     TODO: use different dataloader so i dont have to do this : (
    kdes, pvs = batch
    kdes, pvs = jnp.array(kdes), jnp.array(pvs.unsqueeze(-1))
    
    optimizer, (loss, pred) = train_step(optimizer, kdes, pvs)
    iterator.set_description(f'Loss: {loss.mean()}')
    