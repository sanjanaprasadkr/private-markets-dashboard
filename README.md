# Private Markets Analytics Dashboard

A comprehensive financial analytics dashboard built with Streamlit that provides market analysis, technical indicators, machine learning predictions, and sentiment analysis for stocks.

## Features

- Real-time stock data visualization
- Technical indicators (RSI, MACD, Bollinger Bands, OBV)
- Machine learning price predictions
- Sentiment analysis
- Risk assessment
- Interactive charts and metrics

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/private-markets-dashboard.git
cd private-markets-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Deployment Options

### 1. Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Click "Deploy"

### 2. Heroku

1. Create a `Procfile`:
```
web: streamlit run app.py
```

2. Create a `runtime.txt`:
```
python-3.12.0
```

3. Deploy using Heroku CLI:
```bash
heroku create your-app-name
git push heroku main
```

### 3. Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t private-markets-dashboard .
docker run -p 8501:8501 private-markets-dashboard
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 