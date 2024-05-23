import struct
import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def save_examples_with_labels(images_file, labels_file, output_file, num_examples=10):
    images = read_idx(images_file)
    labels = read_idx(labels_file)

    with open(output_file, 'w') as f:
        for i in range(num_examples):
            image = images[i]
            label = labels[i]
            # Save the flattened pixel values in a C-compatible format
            flattened_pixels = ','.join(str(px) for row in image for px in row)
            f.write(f"float example_{i + 1}[X] = {{{flattened_pixels}}};\n")
            f.write(f"int label_{i + 1} = {label};\n")

# Usage
save_examples_with_labels('data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte', 'mnist_examples.txt', num_examples=10)
