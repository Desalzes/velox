"""Database models for the Archangel AI Monetization System."""
from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum
import uuid

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum as SQLEnum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import sqlalchemy.dialects.sqlite as sqlite

Base = declarative_base()

class SubscriptionTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class UsageType(str, Enum):
    API_REQUEST = "api_request"
    TOKEN_USAGE = "token_usage"
    MODEL_INFERENCE = "model_inference"
    CUSTOM_SERVICE = "custom_service"

class User(Base):
    """User model for customer management."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    subscription_tier = Column(SQLEnum(SubscriptionTier), default=SubscriptionTier.FREE)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # API Access
    api_key = Column(String(64), unique=True, index=True)
    api_key_created_at = Column(DateTime)
    
    # Billing Info
    customer_id = Column(String(255))  # Stripe customer ID
    payment_method_id = Column(String(255))
    
    # Relationships
    usage_records = relationship("UsageRecord", back_populates="user")
    payments = relationship("Payment", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")

class Subscription(Base):
    """Subscription model for recurring billing."""
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tier = Column(SQLEnum(SubscriptionTier), nullable=False)
    status = Column(String(50), default="active")  # active, canceled, expired
    
    # Billing cycle
    starts_at = Column(DateTime, nullable=False)
    ends_at = Column(DateTime, nullable=False)
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    
    # Stripe integration
    stripe_subscription_id = Column(String(255), unique=True)
    stripe_price_id = Column(String(255))
    
    # Usage tracking
    requests_used = Column(Integer, default=0)
    requests_limit = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")

class UsageRecord(Base):
    """Track API usage and costs."""
    __tablename__ = "usage_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Request details
    request_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    
    # AI service details
    ai_provider = Column(String(50))  # openai, anthropic, local
    model_name = Column(String(100))
    usage_type = Column(SQLEnum(UsageType), default=UsageType.API_REQUEST)
    
    # Usage metrics
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Cost tracking
    provider_cost = Column(Float, default=0.0)
    markup_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    # Performance metrics
    response_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Request metadata
    request_metadata = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="usage_records")

class Payment(Base):
    """Payment tracking and billing."""
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Payment details
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    description = Column(Text)
    
    # Status tracking
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING)
    payment_method = Column(String(50))  # card, bank_transfer, etc.
    
    # External payment processor
    stripe_payment_intent_id = Column(String(255), unique=True)
    stripe_charge_id = Column(String(255))
    
    # Billing period
    billing_period_start = Column(DateTime)
    billing_period_end = Column(DateTime)
    
    # Payment metadata
    payment_metadata = Column(JSON)
    failure_reason = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    paid_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="payments")

class AIProvider(Base):
    """AI service provider configuration."""
    __tablename__ = "ai_providers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    base_url = Column(String(255))
    api_key_encrypted = Column(Text)
    is_active = Column(Boolean, default=True)
    
    # Cost configuration
    default_markup_percentage = Column(Float, default=150.0)
    
    # Rate limiting
    requests_per_minute = Column(Integer, default=60)
    tokens_per_minute = Column(Integer, default=1000000)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    models = relationship("AIModel", back_populates="provider")

class AIModel(Base):
    """AI model configuration and pricing."""
    __tablename__ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(Integer, ForeignKey("ai_providers.id"), nullable=False)
    
    name = Column(String(100), nullable=False)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Pricing
    cost_per_input_token = Column(Float, default=0.0)
    cost_per_output_token = Column(Float, default=0.0)
    minimum_cost = Column(Float, default=0.01)
    
    # Capabilities
    max_tokens = Column(Integer, default=4096)
    supports_streaming = Column(Boolean, default=False)
    supports_functions = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    provider = relationship("AIProvider", back_populates="models")

class RateLimitRecord(Base):
    """Rate limiting tracking."""
    __tablename__ = "rate_limit_records"
    
    id = Column(Integer, primary_key=True, index=True)
    identifier = Column(String(255), nullable=False, index=True)  # user_id, ip_address, api_key
    endpoint = Column(String(255), nullable=False)
    
    requests_count = Column(Integer, default=1)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class RevenueAnalytics(Base):
    """Daily revenue analytics aggregation."""
    __tablename__ = "revenue_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # Revenue metrics
    total_revenue = Column(Float, default=0.0)
    subscription_revenue = Column(Float, default=0.0)
    usage_revenue = Column(Float, default=0.0)
    
    # Usage metrics
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_users = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    
    # Cost metrics
    total_provider_costs = Column(Float, default=0.0)
    gross_margin = Column(Float, default=0.0)
    
    # User metrics
    new_users = Column(Integer, default=0)
    churned_users = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)