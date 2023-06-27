# Refactor Notes


## Context

When adapting the TotalSegmentator repository to my neeeds, I found myself in the need to refactor some parts of the code. The existing structure was not intuitive to me, so I ended up changing a lot. I hugely appreciate the work by the original author, and this rewrite is in no way intended a criticism of his work. This is simply the code structure that I found more intuitive to me. Perhaps it may be useful to others as well, or show some ways to simplify the code.

## Changes

The main changes are:
- Create a Task abstraction
  - Task id's, names, weight_url's, class_maps, preview_roi_groups and other parameters were moved from code to configuration files
  - The Task class defines the structure, and the TaskManager class loads the tasks from the configuration files and provides access to them
  - A task may define subtasks to support multi-task models like the default `total`
- Group the various scripts into a CLI interface
- Removed because I don't need it for now, may be restored later:
  - CI scripts I don't need
  - API code
  - tests
- Removed commented-out code
- Added a `settings` class to hold global settings
  - all settings there can be set through the environment
  - the `vars.env` file is loaded automatically for convenience
- Added a `consts` file to hold constants
- Moved from `print` statements to `logging` module
  - obviates the need for the `quiet` and `verbose` flags to the `settings` class
    - reduces the number of parameters passed around
    - same behavior can be achieved by setting the logging level (DEBUG for verbose, WARNING / CRITICAL for quiet)
- Added pre-commit hooks to format code and check for linting errors
- Code now assumes Python 3.10 and a GPU with CUDA (may make safe for Python 3.8 and CPU again later)
- Name changed from `total_segmentator` to `total_segmenter` as I kept misspelling it to that
- `nnUNet` renamed to `nn_unet` to match Python naming conventions
- Added function duration logging, including totals and averages per function, and optionally written to a `.csv` file in the `logs/`
- Added optional W&B logging, mostly to view GPU utilization over time


## Notes

- For the (for me) common usecase of batch inference on a cloud VM with a beefy GPU, the following scripts are added:
  - `./scripts/cuda_install.sh` - installs CUDA and nvidia drivers
  - `./scripts/rsync_ssh.sh` - rsyncs data from a local to a remote directory or vice versa
  - `./scripts/rsync_s3.sh` - rsyncs data from a local directory to an S3 bucket or vice versa
  - `./scripts/shutdown_after.sh` - shuts down the VM after a named process (default: `totalsegmenter`) exits
- The `./scripts/setup.sh` script facilitates setup by:
  - checking for and installing `conda` if not present
  - creating the `totalsegmenter` conda environment from `env.yaml`
  - installing the `pip` packages from `requirements.txt`
  - you can simplify activating the environment by running `source ./scripts/activate.sh`
- After installation, the `totalsegmenter` environment can be activated with `conda activate totalsegmenter`
  - the library is installed in editable mode, so changes to the code are immediately available
  - the `ts` CLI is available, for usage see the `--help` option
- For batch predictions, the following choices are made in the `predict-batch` command:
  - if not provided, the input directory is assumed to be `./data/`
  - all `.nii.gz` files in the input directory are processed recursively
    - the optional `num_splits` and `split_index` facilitate splitting the workload across multiple processes/GPU's: for example, if `num_splits=4` and `split_index=0`, only the first quarter of the files will be processed by this process
    - the `./scripts/predict_batch_multi_gpu.sh` will do this for you
  - outputs are written to `./outputs/{task_name}/{path_relative_to_data_dir}/{file_stem}`
  - the script checks for existing outputs and skips files that already have outputs
    - this currently works for the multi-label file, the statistics and the radiomics file
    - individual labels are not yet checked (TODO)
  - individual labels are written to a subdirectory: `./outputs/{task_name}/{file_stem}/segmentations/{label_name}.nii.gz`
  - the multi-label image is written to `./outputs/{task_name}/{file_stem}/s01.nii.gz`
  - logs are written to `./logs/{timestamp}.log`
- For Docker operations, the following scripts are provided:
  - `./scripts/docker_build.sh` - builds the Docker image
  - `./scripts/docker_push.sh` - pushes the Docker image to Docker Hub
  - `./scripts/docker_run.sh` - runs the Docker image
  - `./scripts/docker_predict.sh` - runs the Docker image with the `predict-batch` command
  - `./scripts/docker_fix_permissions.sh` - fixes permissions on the output directory after running the Docker image (common issue with Docker on Linux)
  - note that docker usage is as of yet untested


## TODO

- [ ] add CuRadiomics for faster radiomics
- [ ] create command to combine masks from multiple tasks for all predictions in outputs/
- [ ] restore API
- [ ] restore tests
- [ ] create script for splitting GPU with MIG based on expected memory usage
- [ ] consider moving preprocessing to `monai` transforms
