"""Revenue engine for processing payments and billing."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from decimal import Decimal
import stripe
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..models.database import User, Subscription, Payment, UsageRecord, PaymentStatus, SubscriptionTier
from ..core.config import get_settings, get_config
from ..core.database import get_db

settings = get_settings()
config = get_config()

# Initialize Stripe
if settings.stripe_secret_key:
    stripe.api_key = settings.stripe_secret_key

class RevenueEngine:
    """Core revenue processing engine."""
    
    def __init__(self):
        self.subscription_tiers = config.get_subscription_tiers()
        self.markup_percentage = settings.default_markup_percentage
        
    async def calculate_usage_cost(
        self, 
        provider_cost: float, 
        usage_type: str = "api_request",
        user_tier: SubscriptionTier = SubscriptionTier.FREE
    ) -> Dict[str, float]:
        """Calculate the cost to charge user based on provider cost."""
        
        # Base markup
        markup_multiplier = self.markup_percentage / 100
        
        # Tier-based discounts
        tier_discounts = {
            SubscriptionTier.FREE: 0.0,
            SubscriptionTier.STARTER: 0.05,  # 5% discount
            SubscriptionTier.PROFESSIONAL: 0.15,  # 15% discount
            SubscriptionTier.ENTERPRISE: 0.25,  # 25% discount
        }
        
        discount = tier_discounts.get(user_tier, 0.0)
        effective_markup = markup_multiplier * (1 - discount)
        
        markup_cost = provider_cost * effective_markup
        total_cost = provider_cost + markup_cost
        
        # Apply minimum charge
        if total_cost < settings.minimum_charge:
            total_cost = settings.minimum_charge
            markup_cost = total_cost - provider_cost
        
        return {
            "provider_cost": round(provider_cost, 6),
            "markup_cost": round(markup_cost, 6),
            "total_cost": round(total_cost, 6),
            "markup_percentage": round(effective_markup * 100, 2),
            "discount_applied": round(discount * 100, 2)
        }
    
    async def record_usage(
        self,
        db: Session,
        user_id: int,
        usage_data: Dict[str, Any]
    ) -> UsageRecord:
        """Record usage and calculate costs."""
        
        # Get user info for tier-based pricing
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Calculate costs
        provider_cost = usage_data.get("provider_cost", 0.0)
        cost_breakdown = await self.calculate_usage_cost(
            provider_cost, 
            usage_data.get("usage_type", "api_request"),
            user.subscription_tier
        )
        
        # Create usage record
        usage_record = UsageRecord(
            user_id=user_id,
            request_id=usage_data.get("request_id"),
            endpoint=usage_data.get("endpoint", ""),
            method=usage_data.get("method", "POST"),
            ai_provider=usage_data.get("ai_provider"),
            model_name=usage_data.get("model_name"),
            usage_type=usage_data.get("usage_type", "api_request"),
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            provider_cost=cost_breakdown["provider_cost"],
            markup_cost=cost_breakdown["markup_cost"],
            total_cost=cost_breakdown["total_cost"],
            response_time_ms=usage_data.get("response_time_ms"),
            success=usage_data.get("success", True),
            error_message=usage_data.get("error_message"),
            metadata=usage_data.get("metadata"),
            ip_address=usage_data.get("ip_address"),
            user_agent=usage_data.get("user_agent")
        )
        
        db.add(usage_record)
        db.commit()
        db.refresh(usage_record)
        
        # Update subscription usage if applicable
        await self._update_subscription_usage(db, user_id, usage_record)
        
        return usage_record
    
    async def _update_subscription_usage(
        self, 
        db: Session, 
        user_id: int, 
        usage_record: UsageRecord
    ):
        """Update subscription request count."""
        active_subscription = db.query(Subscription).filter(
            and_(
                Subscription.user_id == user_id,
                Subscription.status == "active",
                Subscription.current_period_start <= datetime.utcnow(),
                Subscription.current_period_end >= datetime.utcnow()
            )
        ).first()
        
        if active_subscription:
            active_subscription.requests_used += 1
            db.commit()
    
    async def check_usage_limits(self, db: Session, user_id: int) -> Dict[str, Any]:
        """Check if user has exceeded usage limits."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"allowed": False, "reason": "User not found"}
        
        # Get active subscription
        active_subscription = db.query(Subscription).filter(
            and_(
                Subscription.user_id == user_id,
                Subscription.status == "active",
                Subscription.current_period_start <= datetime.utcnow(),
                Subscription.current_period_end >= datetime.utcnow()
            )
        ).first()
        
        if not active_subscription:
            # Check free tier limits
            tier_config = self.subscription_tiers.get("free", {})
            monthly_limit = tier_config.get("request_limit", 100)
            
            # Count requests in current month
            start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            current_usage = db.query(func.count(UsageRecord.id)).filter(
                and_(
                    UsageRecord.user_id == user_id,
                    UsageRecord.created_at >= start_of_month
                )
            ).scalar() or 0
            
            if current_usage >= monthly_limit:
                return {
                    "allowed": False,
                    "reason": "Free tier limit exceeded",
                    "current_usage": current_usage,
                    "limit": monthly_limit
                }
        else:
            # Check subscription limits
            if (active_subscription.requests_limit > 0 and 
                active_subscription.requests_used >= active_subscription.requests_limit):
                return {
                    "allowed": False,
                    "reason": "Subscription limit exceeded",
                    "current_usage": active_subscription.requests_used,
                    "limit": active_subscription.requests_limit
                }
        
        return {"allowed": True}
    
    async def process_subscription_payment(
        self, 
        db: Session, 
        user_id: int, 
        tier: SubscriptionTier
    ) -> Dict[str, Any]:
        """Process subscription payment via Stripe."""
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        tier_config = self.subscription_tiers.get(tier.value, {})
        if not tier_config:
            raise ValueError(f"Invalid subscription tier: {tier}")
        
        monthly_price = tier_config.get("monthly_price", 0)
        
        if monthly_price == 0:
            # Free tier - create subscription without payment
            return await self._create_free_subscription(db, user, tier)
        
        # Create Stripe customer if needed
        if not user.customer_id:
            customer = stripe.Customer.create(
                email=user.email,
                name=user.full_name,
                metadata={"user_id": str(user.id)}
            )
            user.customer_id = customer.id
            db.commit()
        
        # Create Stripe subscription
        try:
            subscription = stripe.Subscription.create(
                customer=user.customer_id,
                items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f'Archangel AI - {tier.value.title()} Plan',
                        },
                        'unit_amount': int(monthly_price * 100),  # Convert to cents
                        'recurring': {
                            'interval': 'month',
                        },
                    },
                }],
                payment_behavior='default_incomplete',
                expand=['latest_invoice.payment_intent'],
            )
            
            # Create local subscription record
            local_subscription = Subscription(
                user_id=user.id,
                tier=tier,
                status="active",
                starts_at=datetime.utcnow(),
                ends_at=datetime.utcnow() + timedelta(days=30),
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30),
                stripe_subscription_id=subscription.id,
                requests_limit=tier_config.get("request_limit", 0)
            )
            
            db.add(local_subscription)
            
            # Update user tier
            user.subscription_tier = tier
            db.commit()
            
            return {
                "success": True,
                "subscription_id": subscription.id,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret,
                "status": subscription.status
            }
            
        except stripe.error.StripeError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_free_subscription(
        self, 
        db: Session, 
        user: User, 
        tier: SubscriptionTier
    ) -> Dict[str, Any]:
        """Create a free tier subscription."""
        
        tier_config = self.subscription_tiers.get(tier.value, {})
        
        # Cancel any existing subscriptions
        existing_subs = db.query(Subscription).filter(
            and_(
                Subscription.user_id == user.id,
                Subscription.status == "active"
            )
        ).all()
        
        for sub in existing_subs:
            sub.status = "canceled"
        
        # Create new subscription
        subscription = Subscription(
            user_id=user.id,
            tier=tier,
            status="active",
            starts_at=datetime.utcnow(),
            ends_at=datetime.utcnow() + timedelta(days=365),  # Free tier for 1 year
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=30),
            requests_limit=tier_config.get("request_limit", 100)
        )
        
        db.add(subscription)
        user.subscription_tier = tier
        db.commit()
        
        return {
            "success": True,
            "subscription_id": subscription.id,
            "status": "active"
        }
    
    async def calculate_monthly_bill(self, db: Session, user_id: int) -> Dict[str, Any]:
        """Calculate monthly bill for pay-per-use customers."""
        
        # Get usage for current month
        start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_of_month = (start_of_month + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        
        usage_records = db.query(UsageRecord).filter(
            and_(
                UsageRecord.user_id == user_id,
                UsageRecord.created_at >= start_of_month,
                UsageRecord.created_at <= end_of_month,
                UsageRecord.success == True
            )
        ).all()
        
        total_cost = sum(record.total_cost for record in usage_records)
        total_requests = len(usage_records)
        total_tokens = sum(record.total_tokens for record in usage_records)
        
        # Group by AI provider
        provider_breakdown = {}
        for record in usage_records:
            provider = record.ai_provider or "unknown"
            if provider not in provider_breakdown:
                provider_breakdown[provider] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            
            provider_breakdown[provider]["requests"] += 1
            provider_breakdown[provider]["tokens"] += record.total_tokens
            provider_breakdown[provider]["cost"] += record.total_cost
        
        return {
            "period": {
                "start": start_of_month.isoformat(),
                "end": end_of_month.isoformat()
            },
            "total_cost": round(total_cost, 2),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "provider_breakdown": provider_breakdown,
            "average_cost_per_request": round(total_cost / max(total_requests, 1), 4)
        }
    
    async def get_revenue_analytics(
        self, 
        db: Session, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get revenue analytics for date range."""
        
        # Total revenue from usage
        usage_revenue = db.query(func.sum(UsageRecord.total_cost)).filter(
            and_(
                UsageRecord.created_at >= start_date,
                UsageRecord.created_at <= end_date,
                UsageRecord.success == True
            )
        ).scalar() or 0.0
        
        # Total revenue from subscriptions
        subscription_revenue = db.query(func.sum(Payment.amount)).filter(
            and_(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date,
                Payment.status == PaymentStatus.COMPLETED
            )
        ).scalar() or 0.0
        
        # Total costs
        provider_costs = db.query(func.sum(UsageRecord.provider_cost)).filter(
            and_(
                UsageRecord.created_at >= start_date,
                UsageRecord.created_at <= end_date,
                UsageRecord.success == True
            )
        ).scalar() or 0.0
        
        # User metrics
        total_users = db.query(func.count(User.id)).scalar() or 0
        active_users = db.query(func.count(func.distinct(UsageRecord.user_id))).filter(
            and_(
                UsageRecord.created_at >= start_date,
                UsageRecord.created_at <= end_date
            )
        ).scalar() or 0
        
        # Request metrics
        total_requests = db.query(func.count(UsageRecord.id)).filter(
            and_(
                UsageRecord.created_at >= start_date,
                UsageRecord.created_at <= end_date,
                UsageRecord.success == True
            )
        ).scalar() or 0
        
        total_revenue = usage_revenue + subscription_revenue
        gross_margin = total_revenue - provider_costs
        margin_percentage = (gross_margin / max(total_revenue, 1)) * 100
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "revenue": {
                "total": round(total_revenue, 2),
                "usage": round(usage_revenue, 2),
                "subscriptions": round(subscription_revenue, 2)
            },
            "costs": {
                "provider_costs": round(provider_costs, 2),
                "gross_margin": round(gross_margin, 2),
                "margin_percentage": round(margin_percentage, 2)
            },
            "users": {
                "total": total_users,
                "active": active_users
            },
            "usage": {
                "total_requests": total_requests,
                "avg_revenue_per_request": round(usage_revenue / max(total_requests, 1), 4)
            }
        }