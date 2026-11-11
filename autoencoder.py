import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from autoencoder_mod import autoenc, FizeauDataset
from astropy.io import fits

from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch


training_frac = 0.8
N = 1001
n_training = int(N*training_frac)
learning_rates = np.array([0.01, 0.001, 0.0001])
batch_size = n_training//4
training_epochs = 100

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = Path("/home/ipa/quanz/user_accounts/efolch/semproj/code/simulated_images_varied_tip_and_tilt_1000/")
out_root = Path("/home/ipa/quanz/user_accounts/efolch/semproj/code/reconstructions_tip_tilt")

for directory in (data_path, out_root):
    if not directory.is_dir():
        directory.mkdir(parents=True)

data = sorted(data_path.glob("*.fits"))
dataset = FizeauDataset(data, log=True)

idxs = np.arange(0, N)
np.random.shuffle(idxs)


train_idxs = idxs[:n_training]
validation_idxs = idxs[n_training:n_training + (N-n_training)//2]
test_idxs = idxs[n_training+(N-n_training)//2:]


train_dataset = Subset(dataset, train_idxs)
validation_dataset = Subset(dataset, validation_idxs)
test_dataset = Subset(dataset, test_idxs)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loss = nn.L1Loss()


loss_lr = np.zeros((3, training_epochs))
models = []

for i, learning_rate in enumerate(learning_rates):
    print(f"learning rate: {learning_rate}")
    time_0 = time.time()
    model = autoenc(1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    loss_per_epoch = np.zeros(training_epochs)
    validation_loss_per_epoch = []
    best_model = model
    
    for epoch in range(training_epochs):
        print(f"Epoch [{epoch + 1}/{training_epochs}]")
        model.train()
        batch_losses = []
        validation_losses = []

        for batch_index, (data, _) in enumerate(tqdm(iter(train_loader))):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)

            # Forward pass: compute the model output
            latent, reconstructed, _, _ = model(data)
            batch_loss = loss(reconstructed, data)
            batch_losses.append(batch_loss.detach().numpy())

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            batch_loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()

        #compute mean loss of current epoch on training set
        loss_per_epoch[epoch] = np.mean(batch_losses)

        model.eval()
        with torch.no_grad():
         
         for val_data in validation_loader:
            val_img, _ = val_data
            _, val_reconstructed, _, _ = model(val_img)
            validation_losses.append(loss(val_reconstructed, val_img))
                
        #compute mean loss of current epoch on validation set
        validation_loss_per_epoch.append(np.mean(validation_losses))

        if epoch >=15:
            if validation_loss_per_epoch[epoch] > np.mean(validation_loss_per_epoch[-15:]):
                break
            else:
               best_model = model

    models.append(best_model) 
    loss_lr[i, :] = np.array(loss_per_epoch)

    time_1 = time.time()

    time_diff_train = time_1 - time_0
    print(f'Elapsed time to train with learning rate {learning_rate:.0e}: {time_diff_train:.2f} seconds')

    subset = test_loader.dataset

for learning_rate, model in zip(learning_rates, models):

    time_0 = time.time()

    model.eval()
    lr_dir = out_root / f"lr_{learning_rate:.0e}"
    if not lr_dir.is_dir():
        lr_dir.mkdir(parents=True)

    with torch.no_grad():
        for i, test_data in enumerate(subset):
            test_img, opd = test_data
            test_img = torch.unsqueeze(test_img, dim=0) #image needs to have shape (1, 1, H, W)
            latent, reconstructed, _, layer_reps = model(test_img)

            src = Path(subset.dataset.files[subset.indices[i]])
            hdr = fits.Header()
            hdr["OPD"] = float(opd)
            hdr["LR"] = float(learning_rate)
            hdr["SOURCE"] = src.name

            fits.writeto(lr_dir / f"{src.stem}_original.fits", test_img.detach().numpy()[0], hdr, overwrite=True)
            fits.writeto(lr_dir / f"{src.stem}_reconstructed.fits", reconstructed.detach().numpy()[0], hdr, overwrite=True)

    print(f"wrote {len(subset)} frame pairs to {lr_dir}")

    time_1 = time.time()
    time_diff_test = time_1 - time_0
    print(f"Elapsed time to test with learning rate {learning_rate:.0e}: {time_diff_test:.2f} seconds")