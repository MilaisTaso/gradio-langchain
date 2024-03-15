run:
	poetry run python src/gradio_app.py
lint:
	poetry run flake8 src
format:
	poetry run black src
sort:
	poetry run isort src