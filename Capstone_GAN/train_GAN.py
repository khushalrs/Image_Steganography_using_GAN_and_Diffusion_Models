import torch
from utils import save_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
from models import Encoder, Decoder, Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MapDataset
from torchvision.utils import save_image

def train_fn(loader, disc, encoder, decoder, opt_enc, opt_dec, opt_disc, mse, bce, beta, gamma):
    loop = tqdm(loader, leave=True)
    for idx, (secret, cover) in enumerate(loop):
        combined = torch.cat((secret,cover),0)
        stego = encoder(combined)
        retrieve = decoder(stego)
        disc_result = disc(stego)

        encoder_mse = mse(stego, cover)
        decoder_mse = mse(retrieve, secret)
        gen_disc_loss = bce(disc_result, torch.ones(stego.size()))
        loss = encoder_mse + beta*decoder_mse + gamma*gen_disc_loss

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        loss.backward()
        opt_enc.step()
        opt_dec.step()

        if idx % 5 == 0:
            disc_cover = disc(cover)
            disc_stego = disc(stego)
            discover = bce(disc_cover, torch.ones(disc_cover.size()))
            disstego = bce(disc_stego, torch.zeros(disc_stego.size()))
            disloss = discover + disstego
            disc.zero_grad()
            disloss.backward()
            opt_disc.step()

def main():
    disc = Discriminator().to(config.DEVICE)
    encoder = Encoder().to(config.DEVICE)
    decoder = Decoder().to(config.DEVICE)
    opt_disc = optim.SGD(disc.parameters(), lr=config.LR_DISCRIMINATOR)
    opt_enc = optim.Adam(encoder.parameters(), lr=config.LR_ENCODER)
    opt_dec = optim.Adam(decoder.parameters(), lr=config.LR_DECODER)
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    disc_scheduler = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=4, gamma=0.9)
    enc_scheduler = torch.optim.lr_scheduler.StepLR(opt_enc, step_size=4, gamma=0.9)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(opt_dec, step_size=4, gamma=0.9)

    #Load dataset
    train_dataset = MapDataset(root_dir_cover=config.TRAIN_DIR_COVER, root_dir_hidden=config.TRAIN_DIR_HIDDEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    for i in range(config.NUM_EPOCHS):
        train_fn(train_loader, disc, encoder, decoder,opt_enc, opt_dec, opt_disc, mse, bce, config.BETA, config.GAMMA)
        disc_scheduler.step()
        enc_scheduler.step()
        dec_scheduler.step()
        if config.SAVE_MODEL and i % 5 == 0:
            save_checkpoint(encoder, opt_enc, filename=config.CHECKPOINT_ENC)
            save_checkpoint(decoder, opt_dec, filename=config.CHECKPOINT_DEC)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()