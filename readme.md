### 初期構築

```bash
docker build -t anaconda .
docker-compose up -d
```

### 分析実行

```bash
cd /analysis_container/src
python analyze.py
```

### 分析結果確認

```bash
python commit_pre_processing.py
```

### 提出ファイル出力

```bash
python commit.py
```
