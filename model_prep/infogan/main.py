from argparse import ArgumentParser

import numpy as np

from easy_inference.train_utils.image_dataset import Dataset

from model_prep.infogan.generator import GeneratorDeconv
from model_prep.infogan.discriminator import Discriminator
from model_prep.infogan.training import Trainer


def train(imgs_dir, img_size, img_ch, batch_size, generator, discriminator,
          save_dir, epochs):
    x_train, x_test = Dataset(img_dir=imgs_dir,
                              fraction_test_set=0,
                              load_resolution=(img_size, img_size, img_ch),
                              val_range=(-1, 1),
                              dtype=np.float32,
                              shuffle=True,
                              max_size=None).load(workers=32)

    if img_ch == 1:
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)

    # x_train = load_mnist("channels_last")

    print("Loaded", len(x_train), x_train.shape)

    generator.model.summary()
    discriminator.model.summary()

    trainer = Trainer(generator, discriminator, save_dir)
    trainer.fit(x_train, num_epochs=1000, batch_size=batch_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--imgs_dir", required=True,
                        help="Path to images")
    parser.add_argument("-g", "--generator", type=str, default=None,
                        help="Path to already existing generator model")
    parser.add_argument("-d", "--discriminator", type=str, default=None,
                        help="Path to already existing discriminator model")
    parser.add_argument("-s", "--save_dir", default="./checkpoints",
                        help="Where to save models during training")
    parser.add_argument('--max_img_size', default=68, type=int,
                        help="Image width == height (only specify for CelebA)")
    parser.add_argument('--black-and-white', action="store_true",
                        help="If this flag is not set, use color")

    # Model parameters
    parser.add_argument('--cont_dim', default=2, type=int,
                        help="Latent continuous dimensions")
    parser.add_argument('--cat_dim', default=10, type=int,
                        help="Latent categorical dimension")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Path to images")
    parser.add_argument('--noise_dim', default=64, type=int,
                        help="noise dimension")
    parser.add_argument('--epochs', default=1000, type=int,
                        help="How many times to go over the dataset")
    args = parser.parse_args()

    img_ch = 1 if args.black_and_white else 3

    if args.generator is None:
        generator = GeneratorDeconv.from_scratch(
            img_size=args.max_img_size,
            img_ch=img_ch,
            cat_dim=args.cat_dim,
            cont_dim=args.cont_dim,
            noise_dim=args.noise_dim,
            batch_size=args.batch_size)
    else:
        generator = GeneratorDeconv.from_path(args.generator)

    if args.discriminator is None:
        discriminator = Discriminator.from_scratch(
            img_size=args.max_img_size,
            img_ch=img_ch,
            cat_dim=args.cat_dim,
            cont_dim=args.cont_dim,
            use_mbd=True)
    else:
        discriminator = Discriminator.from_path(args.discriminator)

    train(imgs_dir=args.imgs_dir,
          batch_size=args.batch_size,
          img_size=args.max_img_size,
          img_ch=img_ch,
          generator=generator,
          discriminator=discriminator,
          save_dir=args.save_dir,
          epochs=args.epochs)
