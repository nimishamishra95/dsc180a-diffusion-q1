# DSC 180A Q1 Project - Diffusion Models

For my Q1 project for the Data Science capstone, I'm exploring diffusion models using the EDM framework from Karras et al. (2022). I'm mainly looking at how classifier-free guidance (CFG) affects image generation and how training with limited data might affect a model's ability to generalize.

For now, I've been working with CIFAR-10 and MNIST. Most of my experiments so far have focused on generating images with pretrained EDM models and comparing how different guidance strengths affect the quality of the generated images.

## Repository Structure

```text
dsc180a-diffusion-q1/
├── notebooks/
│   ├── cifar10_edm_lab.ipynb
│   └── guidance_mnist.ipynb
├── results/
├── .gitignore
└── README.md
```

The `notebooks/` folder has the two starter notebooks I've been using for the project, and `results/` contains the saved outputs from my experiments.

## Setup

I've been running the CIFAR-10 notebook on Google Colab with the T4 GPU, so that's probably the easiest way to reproduce my results.

### Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Go to **File → Open notebook → GitHub**.
3. Paste in the repository URL:

   `https://github.com/nimishamishra95/dsc180a-diffusion-q1`

4. Open `notebooks/cifar10_edm_lab.ipynb`.
5. Go to **Runtime → Change runtime type** and select a T4 GPU or if you have some other GPU then you can select that.
6. Run the notebook cells in order.

The notebook handles most of the setup automatically, including downloading the datasets, pretrained models, and evaluation tools.

### Running Locally

You can also run the notebooks locally if you have Python and the necessary libraries installed.

First, clone the repo:

```bash
git clone https://github.com/nimishamishra95/dsc180a-diffusion-q1.git
cd dsc180a-diffusion-q1
```

I recommend using a separate Python environment so the dependencies don't interfere with anything else.

```bash
conda create -n diffusion python=3.11 -y
conda activate diffusion
```

Install a CUDA-enabled version of PyTorch using the [official installation instructions](https://pytorch.org/get-started/locally/), then install the other main dependencies:

```bash
python -m pip install numpy scipy pandas pyarrow matplotlib pillow jupyterlab ipykernel
```

You can then start JupyterLab:

```bash
jupyter lab
```

The notebooks also use NVIDIA's EDM code, which has some additional dependencies. I've mainly tested the CIFAR-10 experiments in Colab, so I highly recommend you also use the Colab environment for reproducing the current results.

## Data and Models

### CIFAR-10

The CIFAR-10 dataset contains 60000 images across 10 classes, with 50000 used for training and 10000 for testing.

The notebook automatically downloads what it needs including:

- The CIFAR-10 dataset
- Pretrained conditional and unconditional EDM models
- A pretrained CIFAR-10 classifier for checking label accuracy
- An Inception model and reference statistics for calculating FID
- The NVIDIA EDM source code

These files are stored in the `downloads/` folder created by the notebook. The downloaded files are also checked using SHA-256 hashes to make sure they're correct.

If the original CIFAR-10 Python archive is already downloaded, it can be placed at `downloads/cifar-10-python.tar.gz` instead.

I didn't include the downloaded datasets or model weights in the repo since the notebook already handles downloading them.

### MNIST

The MNIST notebook uses a smaller diffusion model for generating handwritten digits and exploring classifier-free guidance.

It downloads the pretrained weights it needs and creates a helper module called `mnist_guidance_lab.py` during setup.

## Reproducing the Results

### CIFAR-10

To reproduce the CIFAR-10 experiments, open `notebooks/cifar10_edm_lab.ipynb` and run all the cells in order.

The notebook has six parts as I mentioned above, starting with unconditional generation and class conditioning, followed by classifier-free guidance comparisons, evaluation using accuracy and FID, and training on smaller subsets of CIFAR-10.

For Part 5, I tested guidance strengths of `0, 1, 1.5, 2, 3, 4`, generating 1,000 images for each setting. Make sure `QUICK = False` before running this part to use the full evaluation configuration.

The generated images and evaluation metrics are saved in the `outputs/` folder created by the notebook. I've also saved the completed Part 5 outputs in the repository's `results/` folder. To reuse the samples I generated without running the generation process again, copy the `.npz` files from `results/` into `outputs/` before running Part 5.

### MNIST

To reproduce the MNIST experiments, just open the `notebooks/guidance_mnist.ipynb` and run the cells from the beginning. The notebook downloads the necessary pretrained weights during setup and then runs the guidance experiments.

## References

- Karras et al. (2022). [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- Ho and Salimans (2022). [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [NVIDIA EDM GitHub Repository](https://github.com/NVlabs/edm)