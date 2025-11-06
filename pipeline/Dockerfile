# Use official Python 3.12 slim base
FROM python:3.12-slim

# install dependencies required for headless chrome
RUN apt-get update && apt-get install -y \
    wget unzip gnupg2 ca-certificates fonts-liberation \
    libnss3 libgconf-2-4 libatk1.0-0 libatk-bridge2.0-0 libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
    xvfb \
 && rm -rf /var/lib/apt/lists/*

# install google-chrome-stable
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
 && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
 && apt-get update && apt-get install -y google-chrome-stable \
 && rm -rf /var/lib/apt/lists/*

# install chromedriver matching the installed chrome
RUN CHROME_VERSION=$(google-chrome --product-version | cut -d '.' -f 1) && \
    CHROMEDRIVER_URL=$(wget -qO- "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_VERSION") && \
    wget -q "https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_URL}/chromedriver_linux64.zip" -O /tmp/chromedriver.zip && \
    unzip /tmp/chromedriver.zip -d /usr/local/bin/ && \
    rm /tmp/chromedriver.zip && chmod +x /usr/local/bin/chromedriver

# set workdir
WORKDIR /app

# copy requirements file if you have one
COPY requirements.txt /app/requirements.txt

# install python deps
RUN pip install --upgrade pip
RUN if [ -f /app/requirements.txt ]; then pip install -r /app/requirements.txt; fi

# copy the repo
COPY . /app

# default command (Render will override with cron command)
CMD ["python", "pipeline/run_realtime_inference.py"]
