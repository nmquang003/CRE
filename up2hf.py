import argparse
import os
import zipfile
from huggingface_hub import login, HfApi

# Hàm để nén các file và thư mục
def zip_files(input_paths, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for input_path in input_paths:
            if os.path.isdir(input_path):
                # Nếu là thư mục, nén toàn bộ thư mục
                for root, dirs, files in os.walk(input_path):
                    for file in files:
                        zipf.write(os.path.join(root, file),
                                   os.path.relpath(os.path.join(root, file), os.path.join(input_path, '..')))
            else:
                # Nếu là file, chỉ nén file đó
                zipf.write(input_path, os.path.basename(input_path))
    print(f"Created zip file: {zip_filename}")

# Hàm để upload lên Hugging Face
def upload_to_hf(zip_filename, target_repo, token):
    # Tách chuỗi theo dấu "/"
    repo_parts = target_repo.rsplit("/")
    repo_id = "/".join(repo_parts[:2])
    target_path = "/".join(repo_parts[2:])
    api = HfApi()
    # Upload the file
    api.upload_file(
        path_or_fileobj=zip_filename,
        path_in_repo=target_path,
        repo_id=repo_id,
        repo_type="dataset"  # Change to "model" or "space" depending on your repo type
    )
    print(f"All inputs uploaded successfully to {target_repo}.")

# Hàm chính để xử lý các tham số đầu vào và gọi các hàm trên
def main():
    # Đọc các tham số đầu vào
    parser = argparse.ArgumentParser(description="Zip and upload files to Hugging Face")
    parser.add_argument('--input', nargs='+', required=True, help="List of files and folders to upload")
    parser.add_argument('--target', required=True, help="Target Hugging Face repo path (user/repo/target_path)")

    args = parser.parse_args()

    # Lấy token từ biến môi trường
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Hugging Face token not found in environment variables. Please set HF_TOKEN.")
        return

    # Tạo tệp zip
    output_dir = os.path.expanduser("~/tmp")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mở rộng đường dẫn cho tmp và tạo tệp zip
    zip_filename = os.path.join(output_dir, "a.zip")

    zip_files(args.input, zip_filename)

    # Upload lên Hugging Face
    upload_to_hf(zip_filename, args.target, token)

if __name__ == "__main__":
    main()