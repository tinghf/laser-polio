import io
import re
import tarfile

import docker
import requests
import sciris as sc


def get_latest_github_tag(repo):
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        latest = response.json()
        return latest["tag_name"].lstrip("v")  # Strip leading 'v'
    else:
        raise RuntimeError(f"Failed to fetch release info: {response.status_code}, {response.text}")


def get_installed_version_from_image(image_name, container_path, package_name):
    client = docker.from_env()
    container = client.containers.create(image=image_name, name="temp_laser", command="sleep 10")

    try:
        tar_stream, stat = container.get_archive(container_path)
        file_like = io.BytesIO(b"".join(tar_stream))
        with tarfile.open(fileobj=file_like) as tar:
            member = tar.getmembers()[0]
            extracted_file = tar.extractfile(member)
            if extracted_file:
                content = extracted_file.read().decode("utf-8")
                match = re.search(rf"{package_name}==([\d\.]+)", content)
                if match:
                    return match.group(1)
                else:
                    raise ValueError(f"Package {package_name} not found in container file.")
            else:
                raise OSError("Failed to extract file from tar.")
    finally:
        container.remove(force=True)


def check_version_match(repo, image_name, container_path, package_name="laser_polio"):
    # print(f"🔍 Checking version for repo: {repo} and image: {image_name}")

    latest_tag = get_latest_github_tag(repo)
    # print(f"📦 Latest GitHub release: {latest_tag}")

    image_version = get_installed_version_from_image(image_name, container_path, package_name)
    # print(f"🐳 Docker image version: {image_version}")

    if latest_tag != image_version:
        # warning_msg = f"⚠️ Version mismatch: GitHub={latest_tag}, Docker={image_version}"
        # warnings.warn(warning_msg, stacklevel=2)
        sc.printred(f"⚠️ Warning: There is a laser_polio version mismatch: GitHub={latest_tag}, Docker={image_version}")
    else:
        sc.printgreen(f"✅ Version match confirmed between GitHub and Docker: {latest_tag}")


# Example usage:
if __name__ == "__main__":
    check_version_match(
        repo="InstituteforDiseaseModeling/laser-polio",
        image_name="idm-docker-staging.packages.idmod.org/laser/laser-polio:latest",
        container_path="/app/laser_polio_deps.txt",
    )


# import io
# import tarfile
# from pathlib import Path

# import docker


# def extract_file_from_image(image_name, container_path, local_path):
#     client = docker.from_env()

#     print(f"Creating temporary container from image: {image_name}")
#     container = client.containers.create(image=image_name, name="temp_laser", command="sleep 10")

#     try:
#         print(f"Extracting file from: {container_path}")
#         tar_stream, stat = container.get_archive(container_path)

#         file_like = io.BytesIO(b"".join(tar_stream))
#         with tarfile.open(fileobj=file_like) as tar:
#             member = tar.getmembers()[0]
#             extracted_file = tar.extractfile(member)
#             if extracted_file:
#                 with open(local_path, "wb") as f:
#                     f.write(extracted_file.read())
#                 local_path_absolute = Path(local_path).resolve()
#                 print(f"✅ File extracted: {local_path_absolute}")
#             else:
#                 print("❌ Failed to extract file.")
#                 return

#     finally:
#         # print("Cleaning up temporary container...")
#         container.remove(force=True)

#     # Simulate `grep laser` output
#     # print(f"\n🔍 Lines containing 'laser' in {local_path}:")
#     with open(local_path) as f:
#         for line in f:
#             if "laser" in line.lower():
#                 print(line.rstrip())


# # Example usage
# extract_file_from_image(
#     image_name="idm-docker-staging.packages.idmod.org/laser/laser-polio:latest",
#     container_path="/app/laser_polio_deps.txt",
#     local_path="./laser_polio_deps.txt",
# )
