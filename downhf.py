import os
import shutil
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

def copy_file_to_output(downloaded_file, output_dir):
    """
    Sao chép file đến thư mục đầu ra.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(downloaded_file))
    shutil.copy(downloaded_file, output_path)
    print(f"Copied file to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download and extract or copy files from Hugging Face")
    parser.add_argument('--repo', required=True, help="Repository ID on Hugging Face (e.g., user/repo)")
    parser.add_argument('--path', required=True, help="Path to the file in the Hugging Face repository")
    parser.add_argument('--output', required=True, help="Directory to save the downloaded file")
    parser.add_argument('--extract', action='store_true', help="Extract the file if it is a ZIP file")
    args = parser.parse_args()

    # Lấy token từ biến môi trường
    token = os.getenv('HF_READ')
    if not token:
        print("Hugging Face read token not found in environment variables. Please set HF_READ.")
        return

    # Tải file từ Hugging Face
    downloaded_file = hf_hub_download(
        repo_id=args.repo,
        filename=args.path,
        repo_type="dataset",
        token=token
    )

    # Kiểm tra và xử lý theo tham số `--extract`
    if args.extract:
        if zipfile.is_zipfile(downloaded_file):
            extract_zip_file(downloaded_file, args.output)
        else:
            print(f"The file {downloaded_file} is not a valid zip file. Cannot extract.")
            return
    else:
        copy_file_to_output(downloaded_file, args.output)

    # Xóa file tạm
    if os.path.exists(downloaded_file):
        os.remove(downloaded_file)
        print(f"Temporary file {downloaded_file} has been deleted.")

if __name__ == "__main__":
    main()
