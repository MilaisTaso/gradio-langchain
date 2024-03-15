import os

from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"), verbose=True
)


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_config() -> Config:
    config = Config()

    # Configクラスの全変数をループしてチェック
    for key, value in vars(config).items():
        if value is None:
            raise ValueError(f"{key} is not set in the environment variables.")

    return config
