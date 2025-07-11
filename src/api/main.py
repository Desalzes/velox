"""Main FastAPI application for Archangel AI Monetization System."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json

from ..core.config import get_settings, get_config
from ..core.database import get_db_session, create_tables
from ..models.database import User, UsageRecord, Subscription
from ..services.ai_gateway import AIGateway
from ..services.revenue_engine import RevenueEngine
from .schemas import *
from .auth import get_current_user, create_access_token, verify_api_key, authenticate_user, create_user, ACCESS_TOKEN_EXPIRE_MINUTES

settings = get_settings()
config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title="Archangel AI Monetization System",
    description="AI-centric monetization platform with multi-provider support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ai_gateway = AIGateway()
revenue_engine = RevenueEngine()
security = HTTPBearer()

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Archangel AI Monetization System",
        "version": "1.0.0",
        "status": "operational",
        "anthropic_configured": bool(settings.anthropic_api_key),
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    ai_health = await ai_gateway.health_check()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "database": "healthy",
            "ai_providers": ai_health
        }
    }

# Authentication Endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db_session)
):
    """Register a new user."""
    try:
        user = create_user(
            db,
            email=user_data.email,
            password=user_data.password,
            username=user_data.username,
            full_name=user_data.full_name,
            is_verified=True  # Auto-verify for simplicity
        )
        
        return UserResponse(
            id=user.id,
            uuid=user.uuid,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            subscription_tier=user.subscription_tier,
            created_at=user.created_at.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login", response_model=Token)
async def login(
    login_data: UserLogin,
    db: Session = Depends(get_db_session)
):
    """Login and get access token."""
    user = authenticate_user(db, login_data.email, login_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, 
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        user=UserResponse(
            id=user.id,
            uuid=user.uuid,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            subscription_tier=user.subscription_tier,
            created_at=user.created_at.isoformat()
        )
    )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        uuid=current_user.uuid,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        subscription_tier=current_user.subscription_tier,
        created_at=current_user.created_at.isoformat()
    )

@app.get("/auth/api-key")
async def get_api_key(current_user: User = Depends(get_current_user)):
    """Get user's API key."""
    return {
        "api_key": current_user.api_key,
        "created_at": current_user.api_key_created_at.isoformat() if current_user.api_key_created_at else None
    }

# AI Chat Completion Endpoint
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """OpenAI-compatible chat completions endpoint."""
    
    # Check usage limits
    usage_check = await revenue_engine.check_usage_limits(db, current_user.id)
    if not usage_check["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Usage limit exceeded",
                "reason": usage_check["reason"],
                "upgrade_url": "/pricing"
            }
        )
    
    # Prepare request data - convert Pydantic objects to dicts
    messages_dict = []
    for msg in request.messages:
        messages_dict.append({
            "role": msg.role,
            "content": msg.content,
            "name": getattr(msg, 'name', None)
        })
    
    request_data = {
        "type": "chat",
        "messages": messages_dict,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream
    }
    
    try:
        # Route request through AI gateway
        result = await ai_gateway.route_request(request.model, request_data)
        
        if request.stream:
            return StreamingResponse(
                stream_chat_response(result, background_tasks, db, current_user),
                media_type="text/plain"
            )
        else:
            # Record usage in background
            background_tasks.add_task(
                record_usage_task,
                db=db,
                user_id=current_user.id,
                result=result,
                endpoint="/v1/chat/completions",
                request_data=request_data
            )
            
            # Return OpenAI-compatible response
            return ChatCompletionResponse(
                id=f"chatcmpl-{generate_id()}",
                object="chat.completion",
                created=int(datetime.utcnow().timestamp()),
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=result["response"]
                        ),
                        finish_reason=result.get("finish_reason", "stop")
                    )
                ],
                usage=Usage(
                    prompt_tokens=result["usage"]["input_tokens"],
                    completion_tokens=result["usage"]["output_tokens"],
                    total_tokens=result["usage"]["total_tokens"]
                )
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat_response(result, background_tasks, db, current_user):
    """Stream chat completion response."""
    if result["success"]:
        # For streaming, we need to handle differently
        # This is a simplified version
        yield f"data: {json.dumps({'choices': [{'delta': {'content': result['response']}}]})}\n\n"
        yield "data: [DONE]\n\n"
    else:
        yield f"data: {json.dumps({'error': result['error']})}\n\n"

# Models endpoint
@app.get("/v1/models")
async def list_models():
    """List available AI models."""
    models = await ai_gateway.get_available_models()
    
    return {
        "object": "list",
        "data": [
            {
                "id": model["id"],
                "object": "model",
                "created": int(datetime.utcnow().timestamp()),
                "owned_by": model["provider"],
                "permission": [],
                "root": model["id"],
                "parent": None
            }
            for model in models
        ]
    }

# Usage tracking endpoint
@app.get("/v1/usage")
async def get_usage(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Get usage statistics for the current user."""
    
    # Parse dates
    if start_date:
        start = datetime.fromisoformat(start_date)
    else:
        start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    if end_date:
        end = datetime.fromisoformat(end_date)
    else:
        end = datetime.utcnow()
    
    # Get usage records
    usage_records = db.query(UsageRecord).filter(
        UsageRecord.user_id == current_user.id,
        UsageRecord.created_at >= start,
        UsageRecord.created_at <= end
    ).all()
    
    # Calculate statistics
    total_requests = len(usage_records)
    total_tokens = sum(record.total_tokens for record in usage_records)
    total_cost = sum(record.total_cost for record in usage_records)
    
    # Group by model
    model_usage = {}
    for record in usage_records:
        model = record.model_name or "unknown"
        if model not in model_usage:
            model_usage[model] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0
            }
        
        model_usage[model]["requests"] += 1
        model_usage[model]["tokens"] += record.total_tokens
        model_usage[model]["cost"] += record.total_cost
    
    return {
        "period": {
            "start": start.isoformat(),
            "end": end.isoformat()
        },
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 4),
        "model_usage": model_usage,
        "subscription_tier": current_user.subscription_tier.value
    }

# Billing endpoint
@app.get("/v1/billing")
async def get_billing_info(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Get billing information for the current user."""
    
    # Get current month's bill
    monthly_bill = await revenue_engine.calculate_monthly_bill(db, current_user.id)
    
    # Get active subscription
    active_subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.status == "active"
    ).first()
    
    subscription_info = None
    if active_subscription:
        subscription_info = {
            "tier": active_subscription.tier.value,
            "status": active_subscription.status,
            "current_period_start": active_subscription.current_period_start.isoformat(),
            "current_period_end": active_subscription.current_period_end.isoformat(),
            "requests_used": active_subscription.requests_used,
            "requests_limit": active_subscription.requests_limit
        }
    
    return {
        "subscription": subscription_info,
        "current_month_usage": monthly_bill,
        "customer_id": current_user.customer_id,
        "payment_method": current_user.payment_method_id is not None
    }

# Subscription management
@app.post("/v1/subscribe")
async def create_subscription(
    request: SubscriptionRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Create or update subscription."""
    
    result = await revenue_engine.process_subscription_payment(
        db, current_user.id, request.tier
    )
    
    if result["success"]:
        return {
            "success": True,
            "subscription_id": result.get("subscription_id"),
            "client_secret": result.get("client_secret"),
            "status": result.get("status")
        }
    else:
        raise HTTPException(status_code=400, detail=result["error"])

# Analytics endpoint (admin only)
@app.get("/v1/analytics")
async def get_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Get system analytics (admin only)."""
    
    # TODO: Add admin role check
    
    if start_date:
        start = datetime.fromisoformat(start_date)
    else:
        start = datetime.utcnow() - timedelta(days=30)
    
    if end_date:
        end = datetime.fromisoformat(end_date)
    else:
        end = datetime.utcnow()
    
    analytics = await revenue_engine.get_revenue_analytics(db, start, end)
    return analytics

# Background task functions
async def record_usage_task(
    db: Session,
    user_id: int,
    result: Dict[str, Any],
    endpoint: str,
    request_data: Dict[str, Any]
):
    """Background task to record usage."""
    
    usage_data = {
        "endpoint": endpoint,
        "method": "POST",
        "ai_provider": result.get("provider"),
        "model_name": result.get("model"),
        "input_tokens": result["cost_info"]["input_tokens"],
        "output_tokens": result["cost_info"]["output_tokens"],
        "total_tokens": result["cost_info"]["total_tokens"],
        "provider_cost": result["cost_info"]["provider_cost"],
        "response_time_ms": result.get("response_time_ms"),
        "success": result.get("success", True),
        "error_message": result.get("error"),
        "metadata": request_data
    }
    
    try:
        await revenue_engine.record_usage(db, user_id, usage_data)
    except Exception as e:
        # Log error but don't fail the request
        print(f"Failed to record usage: {e}")

def generate_id() -> str:
    """Generate a unique ID for responses."""
    import uuid
    return str(uuid.uuid4()).replace("-", "")[:29]

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )