# Digit Recognizer for autogui/chromium bots

## 📂 Python Setup

### Recommend creating your own external python install

```bash
    wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz
    tar xvf Python-3.11.4.tgz
    sudo apt-get update
    sudo apt-get install build-essential libblas-dev liblapack-dev libhdf5-dev libopencv-dev libjpeg-dev libpng-dev libtiff-dev libqt5gui5 libqt5core5a libqt5dbus5 qttools5-dev libgrpc-dev libfreetype6-dev libprotobuf-dev
    cd Python-3.11.4
    ./configure --enable-optimizations
    make
    sudo make install
```

1. Clone the repository:
```bash
   git clone https:/carter4299/bot_num_reader
   cd bot_num_reader
```

2. Create venv:
```bash
    ~/Python-3.11.4/python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
```

3. Install Modules:
```bash
    pip install -r ./config/loose-requirements.txt
```

