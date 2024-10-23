from pydantic import PostgresDsn, computed_field
from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    MODE: str = "dev"

    model_config = SettingsConfigDict(
        env_file=(
            f".env.{os.getenv('MODE')}" if os.getenv("MODE") != "dev" else ".env"
        ),
        env_ignore_empty=True,
        extra="ignore",
    )

    @property
    def mode(self) -> str:
        return self.MODE

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_DB: str

    FTP_HOSTNAME: str
    FTP_PORT: int
    FTP_USERNAME: str
    FTP_PASSWORD: str
    FTP_WORK_DIR: str
    FTP_FILE_PERM: str

    @computed_field
    @property
    def asyncpg_url(self) -> PostgresDsn:
        return MultiHostUrl.build(
            scheme="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            path=self.POSTGRES_DB,
        )

    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_HOST: str
    MYSQL_PORT: int
    MYSQL_DB: str

    @computed_field
    @property
    def mysql_url(self) -> PostgresDsn:
        return MultiHostUrl.build(
            scheme="mysql+asyncmy",
            username=self.MYSQL_USER,
            password=self.MYSQL_PASSWORD,
            host=self.MYSQL_HOST,
            port=self.MYSQL_PORT,
            path=self.MYSQL_DB,
        )

    @property
    def sql_url(self):
        if self.mode == "dev":
            return self.asyncpg_url
        else:
            return self.mysql_url


settings = Settings()
