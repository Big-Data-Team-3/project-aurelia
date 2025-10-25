# services/init_db.py
from sqlalchemy import create_engine, text
from models.user import Base
from config.database import DATABASE_URL, test_connection

def create_tables():
    """Create all database tables"""
    try:
        # Test connection first
        if not test_connection():
            print("❌ Cannot connect to database. Make sure CloudSQL Proxy is running.")
            return False
        
        # Create tables
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        
        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result.fetchall()]
            print(f"📋 Created tables: {tables}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

if __name__ == "__main__":
    create_tables()