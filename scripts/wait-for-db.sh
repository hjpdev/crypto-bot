#!/bin/bash
# wait-for-db.sh

set -e

host="$POSTGRES_HOST"
port="$POSTGRES_PORT"

until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing migrations"
alembic upgrade head

>&2 echo "Starting application"
exec "$@"
