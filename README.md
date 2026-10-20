# AETHER-Q

## Installation

```bash
conda create -n aetherq python=3.11
conda activate aetherq
pip install -r requirements.txt
```

## Run

```bash
python aetherq.py
python aetherq.py --const oneweb
python aetherq.py --src beijing --dst "buenos aires"
python aetherq.py --src "39.9,116.4" --dst "51.5,-0.13"
python aetherq.py --start 2026-10-19T00:00:00 --fmin 0.84
python aetherq.py --headless --hours 24
python aetherq.py --help
```
