# Dockerfile
FROM redis:7.2-alpine

# Optional: Add a custom redis.conf if you want persistence tuning
COPY redis.conf /usr/local/etc/redis/redis.conf

# Default Redis port
EXPOSE 6379

# Run Redis with config (or fallback to defaults)
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]
