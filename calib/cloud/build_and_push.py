import subprocess


def run_docker_commands():
    image_tag = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"
    dockerfile = "calib/Dockerfile"

    build_cmd = ["docker", "build", ".", "-f", dockerfile, "-t", image_tag]
    push_cmd = ["docker", "push", image_tag]

    try:
        subprocess.run(build_cmd, check=True)
        subprocess.run(push_cmd, check=True)
        print("Docker image built and pushed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_docker_commands()
