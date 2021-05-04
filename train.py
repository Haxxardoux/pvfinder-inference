import jax
import jax.numpy as jnp
from flax import optim
import torch
from tqdm import tqdm

from models import UNet, ConvBNrelu

# Initialize dataloader
train_dataset = KDE_to_PV_Dataset('/share/lazy/will/data/June30_2020_80k_1.h5', masking=True, load_xy=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16)

# Initialize model and optimizer
model = UNet(n=16)
parameters = model.init({'params':jax.random.PRNGKey(0)}, jax.numpy.ones((16, 4000, 1)))
optimizer = optim.Adam(learning_rate=3e-4).create(parameters)
optimizer = jax.device_put(optimizer)

# define loss 
def symmetrical_loss(pred, truth):
        valid = ~jnp.isnan(truth)

        # Compute r, only including non-nan values. r will probably be shorter than x and y.
        r = jnp.abs((pred[valid] + self.epsilon) / (truth[valid] + self.epsilon))

        # Compute -log(2r/(r² + 1))
        alpha = -jnp.log(2 * r / (r ** 2 + 1))

        # Sum up the alpha values, and divide by the length of x and y. Note this is not quite
        # a .mean(), since alpha can be a bit shorter than x and y due to masking.
        return alpha.sum() / 4000
    
# define the training step
@jax.jit
def train_step(optimizer, kdes, pvs):
    # defined inside so we can @jax.jit the entire thing instead of the pieces individually. i think...
    def loss_fn(params):
        pred = UNet(n=16).apply({'params': params}, kdes) # i tried to not make this depend on the model (pass UNet(n=16) as argument) but it does not work since this gets compiled 
        return symmetrical_loss(pred, truth), pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    _, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer

for batch in train_loader:
#     TODO: use different dataloader so i dont have to do this : (
    kdes, pvs = batch
    kdes, pvs = jnp.array(batch[0].numpy()), jnp.array(batch[1].numpy())
    
    optimizer = train_step(optimizer, model, kdes, pvs)
    break