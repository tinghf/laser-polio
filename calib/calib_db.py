import os
import platform


def get_storage():
    # 1. Explicit env override always wins
    if os.getenv("STORAGE_URL"):
        storage_url = os.getenv("STORAGE_URL")
        print(f"[INFO] Using STORAGE_URL from environment: {storage_url}")
        return storage_url

    # 2. Windows fallback to SQLite
    if platform.system() == "Windows":
        storage_url = "sqlite:///optuna.db"
        print(f"[INFO] Detected Windows - using local SQLite storage: {storage_url}")
        return storage_url

    # 3. Default to MySQL (Linux, macOS, containers)
    storage_url = "mysql://{}:{}@mysql:3306/{}".format(
        os.getenv("MYSQL_USER", "root"), os.getenv("MYSQL_PASSWORD", ""), os.getenv("MYSQL_DB", "optuna_db")
    )
    print(f"[INFO] Constructed MySQL storage URL: {storage_url}")
    return storage_url
