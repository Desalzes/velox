"""Configuration management for Archangel."""
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseSettings, Field
import yaml

class Settings(BaseSettings):
    """Application settings."""
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    workers: int = Field(default=4, env="WORKERS")
    
    # Railway deployment
    railway_environment: str = Field(default="development", env="RAILWAY_ENVIRONMENT")
    
    # Database
    database_url: str = Field(default="sqlite:///archangel.db", env="DATABASE_URL")
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Security
    jwt_secret: str = Field(env="JWT_SECRET")
    api_key_length: int = Field(default=32)
    password_min_length: int = Field(default=8)
    
    # AI Providers
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Stripe
    stripe_secret_key: Optional[str] = Field(default=None, env="STRIPE_SECRET_KEY")
    stripe_publishable_key: Optional[str] = Field(default=None, env="STRIPE_PUBLISHABLE_KEY")
    stripe_webhook_secret: Optional[str] = Field(default=None, env="STRIPE_WEBHOOK_SECRET")
    
    # Business Configuration
    default_markup_percentage: float = Field(default=150.0)
    minimum_charge: float = Field(default=0.01)
    
    # Rate Limiting
    rate_limiting_enabled: bool = Field(default=True)
    default_rate_limit: str = Field(default="100/minute")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class ConfigLoader:
    """Load configuration from YAML and environment."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return self._substitute_env_vars(config)
        except FileNotFoundError:
            return {}
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables."""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_ai_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get AI provider configuration."""
        return self.get(f"ai_providers.{provider}", {})
    
    def get_subscription_tiers(self) -> Dict[str, Any]:
        """Get subscription tier configuration."""
        return self.get("revenue_models.subscription.tiers", {})
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return self.get("rate_limiting", {})

# Global settings instance
settings = Settings()
config_loader = ConfigLoader()

def get_settings() -> Settings:
    """Get application settings."""
    return settings

def get_config() -> ConfigLoader:
    """Get configuration loader."""
    return config_loader