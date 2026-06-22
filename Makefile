.PHONY: install train train-fast api frontend test docker-up docker-down clean

install:
	pip install -r requirements.txt

train:
	python scripts/train_pipeline.py --model ensemble --build-index

train-fast:
	python scripts/train_pipeline.py --model xgb --build-index

api:
	python scripts/run_api.py

frontend:
	streamlit run frontend/streamlit_app.py

gradio:
	python app.py

test:
	pytest tests/ -v --tb=short

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f models/artifacts/*.joblib models/artifacts/*.json models/artifacts/*.npz
	rm -f data/raw/*.csv data/processed/*.csv data/processed/*.png
