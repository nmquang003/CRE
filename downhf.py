import os
import requests
import zipfile
import argparse

def download_file_from_hf(repo_id, path_in_repo, output_dir, token=None):
    """
    Tải file từ Hugging Face về vị trí cụ thể.
    """
    url = f"https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Tạo đường dẫn file output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(path_in_repo))

    print(f"Downloading {path_in_repo} from {repo_id}...")
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file saved to {output_file}")
    else:
        print(f"Failed to download file: {response.status_code} - {response.text}")
        return None

    return output_file

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
    downloaded_file = download_file_from_hf(args.repo, args.path, args.output, token)

    # Giải nén file nếu cần
    if downloaded_file and args.extract:
        extract_zip_file(downloaded_file, args.output)

if __name__ == "__main__":
    main()
