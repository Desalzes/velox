"""Main entry point for Velox API Monetization System."""
import asyncio
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn
from src.core.config import get_settings
from src.core.database import create_tables, drop_tables
from src.api.main import app

settings = get_settings()

def create_cli_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Velox API Monetization System")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--host", default=settings.host, help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    server_parser.add_argument("--workers", type=int, default=settings.workers, help="Number of workers")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Database commands
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_subparsers = db_parser.add_subparsers(dest="db_command", help="Database commands")
    
    db_subparsers.add_parser("create", help="Create database tables")
    db_subparsers.add_parser("drop", help="Drop database tables")
    db_subparsers.add_parser("reset", help="Drop and recreate database tables")
    
    # User management
    user_parser = subparsers.add_parser("user", help="User management")
    user_subparsers = user_parser.add_subparsers(dest="user_command", help="User commands")
    
    create_user_parser = user_subparsers.add_parser("create", help="Create a new user")
    create_user_parser.add_argument("--email", required=True, help="User email")
    create_user_parser.add_argument("--password", required=True, help="User password")
    create_user_parser.add_argument("--name", help="User full name")
    create_user_parser.add_argument("--admin", action="store_true", help="Make user admin")
    
    # Analytics
    analytics_parser = subparsers.add_parser("analytics", help="Generate analytics report")
    analytics_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    analytics_parser.add_argument("--output", help="Output file for report")
    
    return parser

async def run_server(args):
    """Run the API server."""
    print(f"Starting Velox API Monetization System on {args.host}:{args.port}")
    
    # Auto-create database tables if they don't exist
    try:
        create_tables()
        print("✓ Database tables ensured")
        
        # Create default admin user if no users exist
        from src.core.database import get_db
        from src.api.auth import create_user
        
        with get_db() as db:
            from src.models.database import User
            user_count = db.query(User).count()
            
            if user_count == 0:
                try:
                    # Create admin user with secure password
                    admin_user = create_user(
                        db,
                        email="admin@archangel.ai",
                        password="SecureAdmin2024!",
                        full_name="Default Admin",
                        is_verified=True
                    )
                    print(f"✓ Created default admin user: admin@archangel.ai")
                    print(f"  API Key: {admin_user.api_key}")
                    
                    # Create demo user for public access
                    demo_user = create_user(
                        db,
                        email="demo@archangel.ai", 
                        password="demo123",
                        full_name="Demo User",
                        is_verified=True
                    )
                    print(f"✓ Created demo user: demo@archangel.ai / demo123")
                    print(f"  API Key: {demo_user.api_key}")
                    
                except Exception as e:
                    print(f"⚠️  Could not create default users: {e}")
                    
    except Exception as e:
        print(f"⚠️  Database setup warning: {e}")
    
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        workers=1 if args.reload else args.workers,
        reload=args.reload,
        access_log=True,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()

def handle_db_commands(args):
    """Handle database commands."""
    if args.db_command == "create":
        print("Creating database tables...")
        create_tables()
        print("✓ Database tables created successfully")
        
    elif args.db_command == "drop":
        confirmation = input("Are you sure you want to drop all tables? (y/N): ")
        if confirmation.lower() == 'y':
            print("Dropping database tables...")
            drop_tables()
            print("✓ Database tables dropped successfully")
        else:
            print("Operation cancelled")
            
    elif args.db_command == "reset":
        confirmation = input("Are you sure you want to reset the database? (y/N): ")
        if confirmation.lower() == 'y':
            print("Resetting database...")
            drop_tables()
            create_tables()
            print("✓ Database reset successfully")
        else:
            print("Operation cancelled")

def handle_user_commands(args):
    """Handle user management commands."""
    from src.core.database import get_db
    from src.api.auth import create_user
    
    if args.user_command == "create":
        print(f"Creating user: {args.email}")
        
        try:
            with get_db() as db:
                user = create_user(
                    db,
                    email=args.email,
                    password=args.password,
                    full_name=args.name,
                    is_verified=True
                )
                
                print(f"✓ User created successfully")
                print(f"  ID: {user.id}")
                print(f"  Email: {user.email}")
                print(f"  API Key: {user.api_key}")
                
        except ValueError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def handle_analytics_commands(args):
    """Handle analytics commands."""
    from datetime import datetime, timedelta
    from src.core.database import get_db
    from src.services.revenue_engine import RevenueEngine
    import json
    import asyncio
    
    print(f"Generating analytics report for last {args.days} days...")
    
    revenue_engine = RevenueEngine()
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)
    
    try:
        with get_db() as db:
            analytics = asyncio.run(revenue_engine.get_revenue_analytics(db, start_date, end_date))
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(analytics, f, indent=2, default=str)
                print(f"✓ Analytics report saved to {args.output}")
            else:
                print("\n📊 Analytics Report")
                print("=" * 50)
                print(f"Period: {analytics['period']['start']} to {analytics['period']['end']}")
                print(f"Total Revenue: ${analytics['revenue']['total']:.2f}")
                print(f"  - Usage Revenue: ${analytics['revenue']['usage']:.2f}")
                print(f"  - Subscription Revenue: ${analytics['revenue']['subscriptions']:.2f}")
                print(f"Gross Margin: ${analytics['costs']['gross_margin']:.2f} ({analytics['costs']['margin_percentage']:.1f}%)")
                print(f"Total Users: {analytics['users']['total']}")
                print(f"Active Users: {analytics['users']['active']}")
                print(f"Total Requests: {analytics['usage']['total_requests']}")
                
    except Exception as e:
        print(f"❌ Error generating analytics: {e}")

def show_help():
    """Show help information."""
    print("""
Velox API Monetization System

Available commands:
  server          Start the API server
  db create       Create database tables
  db drop         Drop database tables  
  db reset        Reset database
  user create     Create a new user
  analytics       Generate analytics report

Examples:
  python main.py server --host 0.0.0.0 --port 8000
  python main.py db create
  python main.py user create --email admin@example.com --password secret123
  python main.py analytics --days 7 --output report.json

For more help on specific commands:
  python main.py <command> --help
""")

async def main():
    """Main entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        show_help()
        return
    
    try:
        if args.command == "server":
            await run_server(args)
            
        elif args.command == "db":
            if not args.db_command:
                print("Database command required. Use: create, drop, or reset")
                return
            handle_db_commands(args)
            
        elif args.command == "user":
            if not args.user_command:
                print("User command required. Use: create")
                return
            handle_user_commands(args)
            
        elif args.command == "analytics":
            handle_analytics_commands(args)
            
        else:
            print(f"Unknown command: {args.command}")
            show_help()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        if settings.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())