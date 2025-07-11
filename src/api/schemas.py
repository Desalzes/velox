"""Pydantic schemas for API requests and responses."""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

class SubscriptionTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

# Chat completion schemas
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="The name of the author")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

# User management schemas
class UserCreate(BaseModel):
    email: str = Field(..., description="User email address")
    username: Optional[str] = Field(None, description="Username")
    full_name: Optional[str] = Field(None, description="Full name")
    password: str = Field(..., min_length=8, description="Password")

class UserResponse(BaseModel):
    id: int
    uuid: str
    email: str
    username: Optional[str]
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    subscription_tier: SubscriptionTier
    created_at: str
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

# Subscription schemas
class SubscriptionRequest(BaseModel):
    tier: SubscriptionTier = Field(..., description="Subscription tier")
    payment_method_id: Optional[str] = Field(None, description="Stripe payment method ID")

class SubscriptionResponse(BaseModel):
    id: int
    tier: SubscriptionTier
    status: str
    starts_at: str
    ends_at: str
    current_period_start: str
    current_period_end: str
    requests_used: int
    requests_limit: int
    
    class Config:
        from_attributes = True

# Usage schemas
class UsageRecordResponse(BaseModel):
    id: int
    request_id: str
    endpoint: str
    ai_provider: Optional[str]
    model_name: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    provider_cost: float
    markup_cost: float
    total_cost: float
    response_time_ms: Optional[int]
    success: bool
    created_at: str
    
    class Config:
        from_attributes = True

class UsageSummary(BaseModel):
    period_start: str
    period_end: str
    total_requests: int
    total_tokens: int
    total_cost: float
    successful_requests: int
    failed_requests: int
    average_response_time: Optional[float]
    model_breakdown: Dict[str, Any]

# Payment schemas
class PaymentRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Payment amount")
    payment_method_id: str = Field(..., description="Stripe payment method ID")
    description: Optional[str] = Field(None, description="Payment description")

class PaymentResponse(BaseModel):
    id: int
    amount: float
    currency: str
    status: str
    description: Optional[str]
    created_at: str
    paid_at: Optional[str]
    
    class Config:
        from_attributes = True

# AI Model schemas
class AIModelResponse(BaseModel):
    id: str
    name: str
    provider: str
    type: str
    context_length: int
    description: str
    cost_per_input_token: Optional[float]
    cost_per_output_token: Optional[float]

# Analytics schemas
class RevenueAnalytics(BaseModel):
    period_start: str
    period_end: str
    total_revenue: float
    usage_revenue: float
    subscription_revenue: float
    total_requests: int
    active_users: int
    gross_margin: float
    margin_percentage: float

class UserAnalytics(BaseModel):
    total_users: int
    active_users: int
    new_users: int
    churned_users: int
    users_by_tier: Dict[str, int]
    average_revenue_per_user: float

# Rate limiting schemas
class RateLimitInfo(BaseModel):
    limit: int
    remaining: int
    reset: int
    retry_after: Optional[int] = None

# Error schemas
class ErrorResponse(BaseModel):
    error: str
    message: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Webhook schemas
class StripeWebhookEvent(BaseModel):
    id: str
    object: str
    type: str
    data: Dict[str, Any]
    created: int

# Health check schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]

# Configuration schemas
class PricingTier(BaseModel):
    name: str
    monthly_price: float
    request_limit: int
    rate_limit: str
    features: List[str]

class PricingResponse(BaseModel):
    tiers: List[PricingTier]
    pay_per_use_enabled: bool
    markup_percentage: float
    minimum_charge: float