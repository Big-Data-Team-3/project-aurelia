# services/db_health_service.py
import time
import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from config.database import engine

logger = logging.getLogger(__name__)

class DatabaseHealthService:
    @staticmethod
    async def check_database_health():
        """Check database connection and return health status"""
        try:
            start_time = time.time()
            
            # Test basic connection
            with engine.connect() as connection:
                # Simple query to test connection
                result = connection.execute(text("SELECT 1 as health_check, version() as version"))
                row = result.fetchone()
                
                end_time = time.time()
                connection_time = end_time - start_time
                
                if row and row[0] == 1:
                    return {
                        "status": "healthy",
                        "message": "Database connection successful via CloudSQL Proxy",
                        "database_type": "postgresql",
                        "version": row[1],
                        "connection_time_ms": round(connection_time * 1000, 2),
                        "connection_pool_size": engine.pool.size(),
                        "checked_connections": engine.pool.checkedin()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": "Database query failed",
                        "error": "Unexpected query result"
                    }
                    
        except SQLAlchemyError as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "message": "Database connection failed",
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error during database health check: {str(e)}")
            return {
                "status": "unhealthy",
                "message": "Unexpected database error",
                "error": str(e)
            }