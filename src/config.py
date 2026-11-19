from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = Field(default="lmstudio", validation_alias="LLM_PROVIDER")
    llm_base_url: str = Field(default="", validation_alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    lmstudio_base_url: str = Field(
        default="http://127.0.0.1:1234/v1", validation_alias="LMSTUDIO_BASE_URL"
    )
    lmstudio_api_key: str = Field(default="lm-studio", validation_alias="LMSTUDIO_API_KEY")
    model_name: str = Field(default="qwen2.5-3b-instruct", validation_alias="MODEL_NAME")
    model_version: str = Field(default="local", validation_alias="MODEL_VERSION")
    embedding_model_name: str = Field(
        default="text-embedding-nomic-embed-text-v1.5",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    embedding_model_version: str = Field(default="local", validation_alias="EMBEDDING_MODEL_VERSION")
    embedding_dimension: int = Field(default=384, validation_alias="EMBEDDING_DIMENSION")
    embedding_remote_enabled: bool = Field(default=False, validation_alias="EMBEDDING_REMOTE_ENABLED")
    workspace_root: str = Field(default=".", validation_alias="WORKSPACE_ROOT")
    memory_db_path: str = Field(default=".agent_memory/session.db", validation_alias="MEMORY_DB_PATH")
    default_tenant_id: str = Field(default="default", validation_alias="DEFAULT_TENANT_ID")
    default_brand_id: str = Field(default="default", validation_alias="DEFAULT_BRAND_ID")
    postgres_dsn: str = Field(default="", validation_alias="POSTGRES_DSN")
    max_search_results: int = Field(default=5, validation_alias="MAX_SEARCH_RESULTS")
    lm_timeout_sec: float = Field(default=20.0, validation_alias="LM_TIMEOUT_SEC")
    lm_max_retries: int = Field(default=1, validation_alias="LM_MAX_RETRIES")
    supervisor_model_name: str = Field(default="", validation_alias="SUPERVISOR_MODEL_NAME")
    code_model_name: str = Field(default="", validation_alias="CODE_MODEL_NAME")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8-sig",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def resolved_llm_base_url(self) -> str:
        return self.llm_base_url.strip() or self.lmstudio_base_url.strip()

    @property
    def resolved_llm_api_key(self) -> str:
        return self.llm_api_key.strip() or self.lmstudio_api_key.strip()


settings = Settings()

