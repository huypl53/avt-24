from pydantic import PostgresDsn, computed_field
from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

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
        """
        This is a computed field that generates a PostgresDsn URL for asyncpg.

         The URL is built using the MultiHostUrl.build method, which takes the following parameters:
         - scheme: The scheme of the URL. In this case, it is "postgresql+asyncpg".
         - username: The username for the Postgres database, retrieved from the POSTGRES_USER environment variable.
         - password: The password for the Postgres database, retrieved from the POSTGRES_PASSWORD environment variable.
         - host: The host of the Postgres database, retrieved from the POSTGRES_HOST environment variable.
         - path: The path of the Postgres database, retrieved from the POSTGRES_DB environment variable.

         Returns:
             PostgresDsn: The constructed PostgresDsn URL for asyncpg.
        """
        return MultiHostUrl.build(
            scheme="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            path=self.POSTGRES_DB,
        )


settings = Settings()
