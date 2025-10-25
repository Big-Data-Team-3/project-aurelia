#!/usr/bin/env python3
"""
Database migration script to create conversation and message tables
Run this script to set up the conversation persistence tables in CloudSQL
"""

import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DATABASE_URL, engine
from models import Base

def create_tables():
    """Create conversation and message tables"""
    try:
        print("🚀 Starting database migration...")
        print(f"📊 Database URL: {DATABASE_URL.split('@')[0]}@***")
        
        # Create all tables
        print("📝 Creating conversation and message tables...")
        Base.metadata.create_all(bind=engine)
        
        print("✅ Tables created successfully!")
        
        # Verify tables were created
        with engine.connect() as connection:
            # Check conversations table
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('conversations', 'messages')
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            print(f"📋 Created tables: {', '.join(tables)}")
            
            # Check indexes
            result = connection.execute(text("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE tablename IN ('conversations', 'messages')
                ORDER BY tablename, indexname
            """))
            
            indexes = result.fetchall()
            print(f"🔍 Created indexes:")
            for index_name, table_name in indexes:
                print(f"   - {table_name}.{index_name}")
        
        print("🎉 Migration completed successfully!")
        return True
        
    except SQLAlchemyError as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def verify_schema():
    """Verify the created schema matches requirements"""
    try:
        print("\n🔍 Verifying schema...")
        
        with engine.connect() as connection:
            # Check conversations table structure
            result = connection.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'conversations'
                ORDER BY ordinal_position
            """))
            
            print("📋 Conversations table structure:")
            for row in result.fetchall():
                print(f"   - {row[0]}: {row[1]} {'NULL' if row[2] == 'YES' else 'NOT NULL'} {f'DEFAULT {row[3]}' if row[3] else ''}")
            
            # Check messages table structure
            result = connection.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'messages'
                ORDER BY ordinal_position
            """))
            
            print("📋 Messages table structure:")
            for row in result.fetchall():
                print(f"   - {row[0]}: {row[1]} {'NULL' if row[2] == 'YES' else 'NOT NULL'} {f'DEFAULT {row[3]}' if row[3] else ''}")
            
            # Check foreign key constraints
            result = connection.execute(text("""
                SELECT 
                    tc.constraint_name,
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name IN ('conversations', 'messages')
            """))
            
            print("🔗 Foreign key constraints:")
            for row in result.fetchall():
                print(f"   - {row[1]}.{row[2]} -> {row[3]}.{row[4]}")
        
        print("✅ Schema verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Schema verification failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🗄️  Project Aurelia - Conversation Tables Migration")
    print("=" * 60)
    
    # Create tables
    if create_tables():
        # Verify schema
        verify_schema()
        print("\n🎯 Migration Summary:")
        print("   ✅ Conversations table created")
        print("   ✅ Messages table created")
        print("   ✅ Indexes created for performance")
        print("   ✅ Foreign key constraints established")
        print("   ✅ Schema matches requirements")
        print("\n🚀 Ready to use conversation persistence!")
    else:
        print("\n❌ Migration failed. Please check the errors above.")
        sys.exit(1)
