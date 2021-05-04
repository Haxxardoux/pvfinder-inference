import jax
import torch
import numpy as np
import warnings

# This can throw a warning about float - let's hide it for now.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py



class KDE_to_PV_Dataset(torch.utils.data.Dataset):
    """
    This class collects data. It does not split it up. You can pass in multiple files.
    Example: collect_data('a.h5', 'b.h5')

    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    masking: Turn on or off (default) the masking of hits.
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """

    def __init__(
        self,
        *files,
        dtype=np.float32,
        masking=False,
        slice=None,
        load_xy=False,
        **kargs,
    ):

    
        Xlist = []
        Ylist = []

        print("Loading data...")
        for XY_file in files:
            with h5py.File(XY_file, mode="r") as XY:
                ## .expand_dims(axis=1) makes X (a x b) --> (a x 1 x b) (axis 0, axis 1, axis 2)
                X = np.expand_dims(np.asarray(XY["kernel"]), axis=-1).astype(dtype)
                Y = np.asarray(XY["pv"]).astype(dtype)


                if load_xy:
                    x = np.expand_dims(np.asarray(XY["Xmax"]), axis=-1).astype(dtype)
                    y = np.expand_dims(np.asarray(XY["Ymax"]), axis=-1).astype(dtype)
                    x = x*(X != 0)
                    y = y*(X != 0)
                    X = np.concatenate((X, x, y), axis=-1)  ## filling in axis with (X,x,y)

                if masking:
                    # Set the result to nan if the "other" array is above
                    # threshold and the current array is below threshold
                    Y[(np.asarray(XY["pv_other"]) > 0.01) & (Y < 0.01)] = dtype(np.nan)
#                     Y = jax.ops.index_update(Y, ((np.asarray(XY["pv_other"]) > 0.01) & (Y < 0.01)), jax.numpy.nan)

                Xlist.append(X)
                Ylist.append(Y)

        X = np.concatenate(Xlist, axis=0)
        Y = np.concatenate(Ylist, axis=0)

        if slice:
            X = X[slice, :, :]
            Y = Y[slice, :]
        
        self.X = X
        self.Y = Y
        
        print('Loaded data!')

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.X)

if __name__ == '__main__':
    train_dataset = KDE_to_PV_Dataset('/share/lazy/will/data/June30_2020_80k_1.h5', masking=True, load_xy=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16)
    
    print("Passed test")