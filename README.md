# NeRF Training in Google Colab

This project provides a straightforward, easy-to-use Google Colab notebook for training Neural Radiance Fields (NeRFs) using only a free Google Colab account. It aims to make NeRF experimentation accessible to users without powerful local GPUs.

## Abstract (from the project writeup)

The goal of this project was to modify and compile existing code into one straightforward, easy-to-use notebook compatible with Google Colab. As powerful GPUs are necessary for any ML task, it can be difficult for those with less powerful computers, like laptops, to be able to run ML training. Although solutions to this already exist for UCI students (HPC3), I wanted to create a solution that anyone could use. As Google allows any user to connect to a Tesla T4 GPU, Google Colab was a perfect target for my project. This repository documents the changes made to existing code and the results produced, leveraging free Colab resources.

## Key Features & Modifications

*   **Colab-Native:** Designed to run entirely within the Google Colab environment.
*   **Accessible:** No need for a dedicated Linux machine with an NVIDIA GPU or paid cloud computing.
*   **Pytorch-Based:** Utilizes a Pytorch NeRF implementation ([yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)) for familiarity and ease of modification.
*   **Simplified Training Interface:** The `train()` function is adapted for direct execution in notebook cells, with key parameters exposed and less common ones pre-set.
*   **Google Drive Checkpointing:** Automatically saves model checkpoints to your Google Drive, allowing training to be resumed after disconnections.
*   **Customizable Rendering:**
    *   The number of views to render can now be specified by the user (a limitation in the original rendering script).
    *   Renders views at the input resolution for clearer analysis.
*   **Integrated COLMAP for Pose Estimation:**
    *   Overcame challenges of running COLMAP (which typically uses Docker and GUI elements) within Colab.
    *   Utilizes a non-GUI build of COLMAP from source, thanks to a [contribution by Abbas Salehi](https://github.com/Abbsalehi).
    *   Includes code to process COLMAP outputs into the format required by NeRF (adapted from [LLFF](https://github.com/Fyusion/LLFF)).
*   **Benchmark & Custom Data Support:** Includes clear instructions for using a standard benchmark dataset (Fern) or your own captured images.

## Getting Started

### Prerequisites

1.  A **Google Account** (for Google Colab and Google Drive).
2.  Familiarity with **Google Colab** is helpful but not strictly necessary.
3.  (Optional but Recommended) Git installed on your local machine to clone this repository.

### Setup

1.  **Clone or Download this Repository:**
    *   **Option A (Git):**
        ```bash
        git clone [URL_OF_YOUR_GITHUB_REPOSITORY]
        cd NeRF-Training-in-Google-Colab # Or your repository name
        ```
    *   **Option B (Download ZIP):** Download the ZIP file from GitHub and extract it.

2.  **Upload to Google Drive:**
    *   Upload the entire project folder (e.g., `NeRF-Training-in-Google-Colab`) to the root of your Google Drive (`My Drive`). The notebook expects this location for accessing project files and saving data.
    *   The path in Colab will typically look like `/content/drive/My Drive/NeRF-Training-in-Google-Colab/`.

3.  **Prepare Data (See Notebook for details):**
    *   **Benchmark Data (Fern):** The notebook provides instructions to download, extract, and prepare the Fern dataset. This is recommended for a first run.
    *   **Custom Data:** You can also use your own images. Ensure they are placed in the correct directory structure as outlined in the notebook (`nerf_pytorch/data/your_subject_name/images/`).

## Running the Notebook

1.  Navigate to your Google Drive, find the `NeRF_Training_in_Google_Colab.ipynb` (or your chosen notebook name) file within the uploaded project folder, and open it with Google Colab.
2.  **Ensure GPU is enabled:** Go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator (a T4 GPU is typically provided for free).
3.  **Follow the instructions in each cell of the notebook.** The notebook is designed to be run sequentially. Key steps include:
    *   Mounting Google Drive.
    *   Setting up the data directory (`basedir`).
    *   Running COLMAP for camera pose estimation.
    *   Processing COLMAP results.
    *   Training the NeRF model.
    *   Rendering novel views from the trained NeRF.

## Expected Outputs

*   **Trained NeRF Models:** Saved as checkpoint files (`.tar`) in your Google Drive, within the experiment directory (e.g., `nerf_pytorch/logs/your_experiment_name/`).
*   **Rendered Images/Videos:**
    *   Individual rendered frames (`.png`) are saved during testing and final rendering.
    *   Animated videos (`.mp4`) of novel view syntheses are generated.
    *   These are typically found in a `renderonly_...` or `testset_...` subfolder within your experiment directory.


## Acknowledgements

*   The core NeRF Pytorch implementation is based on [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).
*   The non-GUI COLMAP build solution was adapted from a [pull request by Abbas Salehi](https://github.com/colmap/colmap/pull/950) to the COLMAP repository.
*   Pose processing utilities were adapted from the [LLFF project](https://github.com/Fyusion/LLFF).
*   The original NeRF paper: Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." ECCV 2020.
