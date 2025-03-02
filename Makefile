test:
	python -m pytest ./tests -v

test-cov:
	python -m pytest ./tests --cov=tests --cov-report=term-missing -v

test-cov-html:
	python -m pytest ./tests --cov=tests --cov-report=html -v

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements.test.txt

clean:
	docker-compose down -v
	rm -rf data/postgres/* data/exports/* __pycache__ .pytest_cache .coverage htmlcov build dist

lint:
	python -m flake8 --max-line-length=100 ./app ./tests
	python -m black ./app ./tests --check

format:
	python -m black ./app ./tests

run:
	python -m app.main

.PHONY: build up down logs migrate

build:
	docker-compose build

start:
	./scripts/wait-for-db.sh
	python -m app.main

up:
	docker-compose up --build

dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

down:
	docker-compose down --remove-orphans

logs:
	docker-compose logs -f

migrate:
	docker-compose exec app alembic upgrade head
