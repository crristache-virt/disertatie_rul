# Prediction of Hardware Failures and Remaining Useful Life (RUL) in Hard Disk Drives using SMART data and Machine Learning

## Project Goal
This project aims to analyze the Backblaze SMART HDD dataset and build machine learning models to predict disk failure risk and estimate Remaining Useful Life (RUL) for hard disk drives. The pipeline is modular and designed for research experimentation.

## Dataset Source
- [Backblaze Hard Drive Data](https://www.backblaze.com/b2/hard-drive-test-data.html)
- Dataset contains SMART statistics for thousands of HDDs, including failure events.

## Setting DATA_DIR
- By default, the code expects data in the `data/` folder.
- You can override this by setting the `DATA_DIR` environment variable to the path containing your Backblaze dataset files.

## Running Notebooks
- All notebooks are in the `notebooks/` folder.
- Notebooks will print instructions if the dataset is missing (no automatic download).
- Run notebooks in order for a complete pipeline demonstration.

## Repository Structure
- `src/`: Modular pipeline code (config, io, cleaning, preprocessing, features, labeling, split, models, evaluation, utils)
- `data/`: Place Backblaze SMART dataset files here
- `notebooks/`: Step-by-step analysis and modeling
- `reports/figures/`: Output figures and reports
- `.devcontainer/`: Development container setup
