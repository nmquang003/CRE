import os
from huggingface_hub import hf_hub_download
import zipfile
import argparse

def extract_zip_file(zip_file, extract_to):
    """
    Giải nén file ZIP vào thư mục chỉ định.
    """
    if not zipfile.is_zipfile(zip_file):
        print(f"The file {zip_file} is not a valid zip file.")
        return

    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_file} to {extract_to}")

def main():
    parser = argparse.ArgumentParser(description="Download and extract files from Hugging Face")
    parser.add_argument('--repo', required=True, help="Repository ID on Hugging Face (e.g., user/repo)")
    parser.add_argument('--path', required=True, help="Path to the file in the Hugging Face repository")
    parser.add_argument('--output', required=True, help="Directory to save the downloaded file")
    parser.add_argument('--extract', action='store_true', help="Extract the file if it is a ZIP file")
    args = parser.parse_args()

    # Tải file từ Hugging Face
    token = os.getenv('HF_READ')
    if not token:
        print("Hugging Face write token not found in environment variables. Please set HF_WRITE.")
        return

    downloaded_file = hf_hub_download(
        repo_id=args.repo,
        filename=args.path,
        repo_type="dataset",
        token=token
    )

    # Giải nén file nếu cần
    extract_zip_file(downloaded_file, args.output)
    if os.path.exists(downloaded_file):
        os.remove(downloaded_file)
        print(f"Temporary ZIP file {downloaded_file} has been deleted.")

if __name__ == "__main__":
    main()
