from utils import mnist_loader
import numpy as np
from DimReductionMIssingData import DimReductionMissingData
import matplotlib.pyplot as plt

def set_random_missing_values(data, fraction=0.1, value=np.nan):
    assert(len(data.shape) == 2)
    num_samples, num_cols = data.shape
    row_indices, col_indices = range(num_samples), range(num_cols)
    num_columns_to_miss = int(fraction * num_cols)
    for i in xrange(num_samples):
        np.random.shuffle(col_indices)
        data[i, col_indices[0:num_columns_to_miss]] = value

def show_mnist_image(data_org, data_missing, data_reconstructed):
    num_images = 10
    c_map = plt.cm.gray
    c_map.set_bad('r', 1.)

    def clip_img(img_r):
        img_r[img_r < 0.2*(np.max(img_r) - np.min(img_r))] = 0
        return img_r

    plt.figure(figsize=(9, 3), frameon=False)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    for i in xrange(num_images):
        idx = np.random.randint(low=0, high=data_missing.shape[0])
        for j in range(3):
            img = {
                0: data_org[idx, :].reshape(28, 28),
                1: data_missing[idx, :].reshape(28, 28),
                2: clip_img(data_reconstructed[idx, :]).reshape(28, 28)
            }[j]
            plt.subplot(3, num_images, j * num_images + i + 1)
            plt.subplot(3, num_images, j * num_images + i + 1).axes.get_xaxis().set_visible(False)
            plt.subplot(3, num_images, j * num_images + i + 1).axes.get_yaxis().set_visible(False)
            msk = np.ma.array(img, mask=np.isnan(img))
            plt.imshow(msk, interpolation='nearest', cmap=c_map)
    plt.show()

def main():
    data_set, _, _ = mnist_loader()
    data = np.asarray(data_set[0][0:20000])
    np.random.shuffle(data)
    data_org = data.copy()
    set_random_missing_values(data, 0.70)
    dim_red_object = DimReductionMissingData(data, reduced_dim=60)
    B = dim_red_object.optimize(num_epochs=5, batch_size=1000)
    data_reduced = dim_red_object.get_reduced_dimensions(B=B, X=dim_red_object.X,  mask=dim_red_object.X_mask)
    data_reconstructed = np.dot(data_reduced, B) + dim_red_object.X_mean
    show_mnist_image(data_org, data, data_reconstructed)

if __name__ == '__main__':
    main()
