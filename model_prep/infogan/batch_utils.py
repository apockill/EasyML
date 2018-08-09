import numpy as np


def sample_noise(noise_scale, batch_size, noise_dim):
    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim))


def sample_cat(batch_size, cat_dim):
    y = np.zeros((batch_size, cat_dim), dtype="float32")
    random_y = np.random.randint(0, cat_dim, size=batch_size)
    y[np.arange(batch_size), random_y] = 1
    return y


def get_gen_batch(batch_size, cat_dim, cont_dim, noise_dim, noise_scale):
    x_gen = sample_noise(noise_scale, batch_size, noise_dim)
    y_gen = np.zeros((x_gen.shape[0], 2), dtype=np.uint8)
    y_gen[:, 1] = 1

    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)

    # Repeat y_cont to accomodate for keras" loss function conventions
    y_cont_target = np.expand_dims(y_cont, 1)
    y_cont_target = np.repeat(y_cont_target, 2, axis=1)

    return x_gen, y_gen, y_cat, y_cont, y_cont_target


def get_real_disc_batch(real_batch, batch_size, noise_scale, cat_dim, cont_dim,
                        label_smoothing=False, label_flipping=0):
    y_disc = np.zeros((real_batch.shape[0], 2), dtype=np.uint8)
    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)

    # Produce an output
    y_disc[:, 1] = 1

    if label_smoothing:
        y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
    else:
        y_disc[:, 1] = 1

    if label_flipping > 0:
        p = np.random.binomial(1, label_flipping)
        if p > 0:
            y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Repeat y_cont to accomodate for keras" loss function conventions
    y_cont = np.expand_dims(y_cont, 1)
    y_cont = np.repeat(y_cont, 2, axis=1)

    return real_batch, y_disc, y_cat, y_cont


def get_fake_disc_batch(generator_model, batch_size, cat_dim, cont_dim,
                        noise_dim, noise_scale, label_flipping=0):
    # Pass noise to the generator
    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)

    # Produce an output
    fake_batch = generator_model.predict([y_cat, y_cont, noise_input],
                                         batch_size=batch_size)
    y_disc = np.zeros((fake_batch.shape[0], 2), dtype=np.uint8)
    y_disc[:, 0] = 1

    if label_flipping > 0:
        p = np.random.binomial(1, label_flipping)
        if p > 0:
            y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Repeat y_cont to accomodate for keras" loss function conventions
    y_cont = np.expand_dims(y_cont, 1)
    y_cont = np.repeat(y_cont, 2, axis=1)

    return fake_batch, y_disc, y_cat, y_cont