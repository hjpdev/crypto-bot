#!/bin/bash
# wait-for-db.sh

set -e

# host="db"
host="localhost"
port="5432"

until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "postgres" -d "crypto_trader" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing migrations"
alembic upgrade head

>&2 echo "Starting application"
exec "$@"
