#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Use PYTHON variable if set, else default to 'python3'
PYTHON_COMMAND=${PYTHON:-python3}

# Initialize variables
EXTRA_URL=""
NO_ROOT=0
KEEP_VENV=0
WITHOUT=""
WITH=""

# Parse command-line arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --no-root)
    NO_ROOT=1
    shift
    ;;
    --keep-venv)
    KEEP_VENV=1
    shift
    ;;
    --without)
    WITHOUT="$2"
    shift # past argument
    shift # past value
    ;;
    --with)
    WITH="$2"
    shift # past argument
    shift # past value
    ;;
    *)
    if [[ -z "$EXTRA_URL" ]]; then
        EXTRA_URL="$1"
        shift
    else
        echo "Unknown argument: $1"
        echo "Usage: $0 [--no-root] [--keep-venv] [--without <value>] [--with <value>] <extra_torch_url_py>"
        exit 1
    fi
    ;;
esac
done

# Check if EXTRA_URL is set
if [ -z "$EXTRA_URL" ]; then
    echo "Usage: $0 [--no-root] [--keep-venv] [--without <value>] [--with <value>] <extra_torch_url_py>"
    exit 1
fi

# Step 1: Build the poetry export command with optional arguments
POETRY_EXPORT_CMD=(poetry export --without-hashes --format=requirements.txt --extras "${EXTRAS}")

if [[ -n "$WITHOUT" ]]; then
    POETRY_EXPORT_CMD+=("--without" "$WITHOUT")
fi

if [[ -n "$WITH" ]]; then
    POETRY_EXPORT_CMD+=("--with" "$WITH")
fi

# Export the dependencies to requirements.txt without hashes
"${POETRY_EXPORT_CMD[@]}" > requirements.txt

# Step 2: Delete the existing virtual environment (--keep-venv reuses it, e.g. restored from a CI cache)
if [ -d "./.venv" ] && [ $KEEP_VENV -eq 0 ]; then
    rm -rf ./.venv
fi

# Step 3: Create a new virtual environment using the specified Python command
if [ ! -d "./.venv" ]; then
    $PYTHON_COMMAND -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Step 4: Extract the locked versions of torch and torchvision from requirements.txt
TORCH_VERSION=$(grep '^torch==' requirements.txt | awk -F'==' '{print $2}' | awk '{print $1}')
TORCHVISION_VERSION=$(grep '^torchvision==' requirements.txt | awk -F'==' '{print $2}' | awk '{print $1}')

# Check if torch version was found
if [ -z "$TORCH_VERSION" ]; then
    echo "Torch version not found in requirements.txt"
    exit 1
fi
TORCH_SPEC="torch==${TORCH_VERSION}"
if [ -n "$TORCHVISION_VERSION" ]; then
    TORCH_SPEC="$TORCH_SPEC torchvision==${TORCHVISION_VERSION}"
else
    TORCH_SPEC="$TORCH_SPEC torchvision"
fi

# Remove lines containing 'torch', 'nvidia', 'cuda' and 'triton' from requirements.txt
# torch >= 2.10 depends on the cuda-toolkit meta package instead of nvidia-* wheels
sed -i '/^torch==/d' requirements.txt
sed -i '/torchvision/d' requirements.txt
sed -i '/nvidia/d' requirements.txt
sed -i '/^cuda-/d' requirements.txt
sed -i '/triton/d' requirements.txt

# Step 5: Install torch with the extracted version and other dependencies

python -m pip install -r requirements.txt --no-cache-dir --extra-index-url "$EXTRA_URL"
python -m pip list --format=freeze | grep nvidia | xargs python -m  pip uninstall -y triton torch torchvision
# shellcheck disable=SC2086
python -m pip install $TORCH_SPEC --index-url "$EXTRA_URL" --no-cache-dir

# Step 6: Optionally install the current package
if [[ $NO_ROOT -eq 0 ]]; then
    python -m pip install -e . --no-deps --no-cache-dir --extra-index-url "$EXTRA_URL"
fi

# rm requirements.txt

echo "Script executed successfully!"