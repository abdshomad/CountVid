# CountVid: Installation Guide with UV Venv

This guide provides step-by-step installation instructions for CountVid using `uv venv` as the virtual environment manager instead of conda.

## Prerequisites

### 1. Install GCC

Install GCC. In this project, GCC 11.3 and 11.4 were tested. The following command installs GCC and other development libraries and tools required for compiling software in Ubuntu.

```bash
sudo apt update
sudo apt install build-essential
sudo apt install gcc-11 g++-11
```

### 2. Install CUDA Toolkit

NOTE: In order to install detectron2 in step 4, you need to install CUDA Toolkit. Refer to: https://developer.nvidia.com/cuda-downloads

### 3. Install UV

Install uv (Python package manager) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
pip install uv
```

## Installation Steps

### 1. Clone Repository

```bash
git clone git@github.com:niki-amini-naieni/CountVid.git
cd CountVid
```

### 2. Create Virtual Environment

Create a virtual environment with Python 3.10 using uv:

```bash
uv venv --python 3.10
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### 3. Install System Dependencies

Install required compilers and ensure GCC 11 is being used:

```bash
# Install required compilers (if not already installed)
sudo apt install gxx-11

# Set environment variable to ensure gcc 11 is used for compilation
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```

### 4. Install SAM2

Clone and install SAM2:

```bash
cd ..
git clone https://github.com/facebookresearch/sam2.git
cd sam2
uv pip install -e .
cd ../CountVid
```

### 5. Install Python Dependencies

Install requirements using uv:

```bash
uv pip install -r requirements.txt
```

### 6. Build GroundingDINO Operations

Build the GroundingDINO operations:

```bash
# Install torch and setuptools first (required for building)
uv pip install torch torchvision setuptools

# Build the operations with no build isolation
cd models/GroundingDINO/ops
uv pip install -e . --no-build-isolation
python test.py  # should result in 6 lines of * True
cd ../../../
```

### 7. Install Detectron2

Install detectron2:

```bash
uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

**Note**: The `--no-build-isolation` flag is required because detectron2's setup.py requires `torch` to be available during the build process. This flag allows the build process to use the already installed packages in the virtual environment instead of creating an isolated build environment.

### 8. Download Pre-Trained Weights

Create the checkpoints directory:

```bash
mkdir checkpoints
```

Install gdown for downloading models from Google Drive:

```bash
uv pip install gdown
```

Download BERT weights:

```bash
python download_bert.py
```

Download the pretrained CountGD-Box model:

```bash
gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/
```

Download the pretrained SAM 2.1 weights:

```bash
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 8.1. Retry Failed Downloads

If any of the model downloads fail, you can retry them individually:

**Retry BERT weights:**
```bash
python download_bert.py
```

**Retry CountGD-Box model:**
```bash
gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/
```

**Retry SAM 2.1 weights:**
```bash
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

**Verify downloads:**
```bash
ls -la checkpoints/
du -h checkpoints/*
```

**Expected file sizes:**
- `bert-base-uncased/` directory: ~417MB
- `countgd_box.pth`: ~911MB  
- `sam2.1_hiera_large.pt`: ~857MB

## Demo Setup and Execution

### 1. Download Demo Frames

Create the demo directory and download video frames:

```bash
mkdir demo
```

Download the video frames for the demo from [here](https://drive.google.com/drive/folders/1v4RNNBHYEQQ82NF8fNiRPhIdQ96-7xCs?usp=sharing), and place them into the `demo` directory, so your file tree looks like:

```
CountVid/
  |demo/
    |00001.jpg
    ...
    |00094.jpg
  ...
```

### 2. Run Demo

Run the following command:

```bash
python count_in_videos.py --video_dir demo --input_text "penguin" --sam_checkpoint checkpoints/sam2.1_hiera_large.pt --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml --obj_batch_size 5 --img_batch_size 2 --downsample_factor 2 --pretrain_model_path checkpoints/countgd_box.pth --temp_dir ./demo_temp --output_dir ./demo_output --save_final_video --save_countgd_video
```

**Note**: The batch sizes have been reduced from the original values to prevent CUDA out of memory errors:
- `--obj_batch_size 5` (reduced from 30)
- `--img_batch_size 2` (reduced from 10) 
- `--downsample_factor 2` (increased from 1 to reduce memory usage)

If you have more GPU memory available, you can try increasing these values gradually.

### 3. Visualize Output

You should see the following videos saved to the `demo_output` folder once the demo has finished running:

- `final-video.mp4` - Final output video with counting results
- `countgd-video.avi` - Timelapse boxes from CountGD-Box

## Environment Management

### Activating the Environment

To activate the virtual environment in future sessions:

```bash
cd CountVid
source .venv/bin/activate
```

### Deactivating the Environment

To deactivate the virtual environment:

```bash
deactivate
```

### Updating Dependencies

To update dependencies:

```bash
uv pip install --upgrade -r requirements.txt
```

## Troubleshooting

### Build Isolation Issues

If you encounter errors like `ModuleNotFoundError: No module named 'torch'` during package installation, this is likely due to build isolation. Some packages (like detectron2 and GroundingDINO operations) require dependencies to be available during the build process.

**Solution**: Use the `--no-build-isolation` flag:

```bash
uv pip install package_name --no-build-isolation
```

**Impact of `--no-build-isolation`**:
- **What it does**: Allows the build process to access packages already installed in your virtual environment
- **When to use**: Required for packages whose setup.py imports dependencies during build time
- **Trade-off**: Slightly less isolated build environment, but necessary for complex packages
- **Security**: Generally safe when using a clean virtual environment

### GCC Compilation Issues

If you encounter compilation issues, ensure GCC 11 is being used:

```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```

### CUDA Issues

Make sure CUDA toolkit is properly installed and accessible:

```bash
nvcc --version
```

### Memory Issues

If you encounter memory issues during installation, you can try:

```bash
export MAX_JOBS=1
```

### CUDA Out of Memory (OOM) Issues

If you encounter `torch.cuda.OutOfMemoryError` when running the demo, try these solutions:

**Reduce batch sizes:**
```bash
# Start with very small batch sizes
python count_in_videos.py --video_dir demo --input_text "penguin" --sam_checkpoint checkpoints/sam2.1_hiera_large.pt --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml --obj_batch_size 1 --img_batch_size 1 --downsample_factor 4 --pretrain_model_path checkpoints/countgd_box.pth --temp_dir ./demo_temp --output_dir ./demo_output --save_final_video --save_countgd_video
```

**Memory optimization parameters:**
- `--obj_batch_size`: Reduce from 5 to 1-3
- `--img_batch_size`: Reduce from 2 to 1
- `--downsample_factor`: Increase from 2 to 4 or higher
- Add `--no_save_intermediate` to reduce memory usage

**Check GPU memory:**
```bash
nvidia-smi
```

**Clear GPU memory before running:**
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Alternative: Use CPU (slower but uses less memory):**
```bash
# Add --device cpu to force CPU usage
python count_in_videos.py --device cpu --video_dir demo --input_text "penguin" ...
```

### Download Issues

If model downloads fail, try these solutions:

**Network connectivity issues:**
```bash
# Test basic connectivity
ping -c 3 google.com
ping -c 3 8.8.8.8

# Test DNS resolution
nslookup dl.fbaipublicfiles.com
nslookup drive.google.com
```

**Retry with different methods:**
```bash
# For SAM 2.1 weights, try curl instead of wget
curl -L -o checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# For CountGD-Box, try with different gdown options
gdown 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/countgd_box.pth
```

**Partial download recovery:**
```bash
# Check if files are partially downloaded
ls -la checkpoints/

# Remove incomplete files and retry
rm checkpoints/incomplete_file.pth
# Then retry the download
```

**Alternative download locations:**
- **CountGD-Box**: If Google Drive fails, check the original paper repository for alternative download links
- **SAM 2.1**: If Facebook's CDN fails, try the official SAM2 repository releases
- **BERT**: The download_bert.py script should handle retries automatically

## Next Steps

After successful installation, you can:

1. **Run the demo** as described above
2. **Download datasets** following the instructions in the main README.md
3. **Reproduce paper results** using the commands in the main README.md
4. **Train CountGD-Box** using the training code available [here](https://drive.google.com/file/d/1jLe9OP4MXr-yVfS-CruXRDF6D9H2Bi7-/view?usp=sharing)

## Notes

- This installation guide replaces the conda-based installation from the original README
- All functionality remains the same, only the virtual environment management has changed
- The original README.md still contains all the dataset download and evaluation instructions
- For any issues, refer to the original README.md or contact the authors at [niki.amini-naieni@eng.ox.ac.uk](mailto:niki.amini-naieni@eng.ox.ac.uk)
