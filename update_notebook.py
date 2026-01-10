import json
import os

nb_path = r"d:\KLTN\TranSTR\causalvid\transtr.ipynb"

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Create new cell
    new_cell_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Setup: Download Data from Hugging Face\n",
            "Use the cell below to download and extract your object features."
        ]
    }

    # Using 'resolve' instead of 'blob' for direct download
    hf_url = "https://huggingface.co/datasets/DanielQ07/kltn/resolve/main/objbox.tar.gz"

    new_cell_code = {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
           "# --- Hugging Face Download Configuration ---\n",
           f"HF_FILE_URL = \"{hf_url}\" \n",
           "HF_REPO_ID = \"\"   \n",
           "HF_FILENAME = \"objbox.tar.gz\"\n",
           "\n",
           "import os\n",
           "import tarfile\n",
           "import urllib.request\n",
           "\n",
           "# Define Output Directory (Kaggle Working Directory or Local)\n",
           "OUTPUT_DIR = \"/kaggle/working/features/objects\" \n",
           "if not os.path.exists(\"/kaggle/working\"):\n",
           "    # Local fallback: d:\\KLTN\\TranSTR\\causalvid\\features\\objects\n",
           "    OUTPUT_DIR = os.path.join(os.getcwd(), \"features\", \"objects\")\n",
           "\n",
           "if not os.path.exists(OUTPUT_DIR):\n",
           "    os.makedirs(OUTPUT_DIR)\n",
           "\n",
           "print(f\"Target Directory: {OUTPUT_DIR}\")\n",
           "\n",
           "if HF_FILE_URL:\n",
           "    print(f\"Downloading from {HF_FILE_URL}...\")\n",
           "    tar_path = os.path.join(OUTPUT_DIR, \"objbox.tar.gz\")\n",
           "    \n",
           "    try:\n",
           "        # Simple download \n",
           "        urllib.request.urlretrieve(HF_FILE_URL, tar_path)\n",
           "        print(\"Download complete.\")\n",
           "        \n",
           "        if os.path.exists(tar_path):\n",
           "            print(\"Extracting...\")\n",
           "            with tarfile.open(tar_path, \"r:gz\") as tar:\n",
           "                # Helper to prevent unsafe extraction\n",
           "                def is_within_directory(directory, target):\n",
           "                    abs_directory = os.path.abspath(directory)\n",
           "                    abs_target = os.path.abspath(target)\n",
           "                    prefix = os.path.commonprefix([abs_directory, abs_target])\n",
           "                    return prefix == abs_directory\n",
           "                \n",
           "                def safe_extract(tar, path=\".\", members=None, *, numeric_owner=False):\n",
           "                    for member in tar.getmembers():\n",
           "                        member_path = os.path.join(path, member.name)\n",
           "                        if not is_within_directory(path, member_path):\n",
           "                            raise Exception(\"Attempted Path Traversal in Tar File\")\n",
           "                    tar.extractall(path, members, numeric_owner=numeric_owner) \n",
           "                    \n",
           "                safe_extract(tar, path=OUTPUT_DIR)\n",
           "            print(\"Extraction Done.\")\n",
           "            \n",
           "            # Verify what was extracted\n",
           "            print(f\"Contents of {OUTPUT_DIR}: {os.listdir(OUTPUT_DIR)[:10]}\")\n",
           "            \n",
           "    except Exception as e:\n",
           "        print(f\"Error during processing: {e}\")\n"
       ]
    }

    # Insert at index 1 (after imports)
    if 'cells' in nb:
        nb['cells'].insert(1, new_cell_markdown)
        nb['cells'].insert(2, new_cell_code)
        
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated successfully via JSON.")
    else:
        print("Error: Invalid notebook format.")

except Exception as e:
    print(f"FAILED: {e}")
