# 初期構築
docker build -t anaconda .
docker-compose up -d

# 分析実行
cd /analysis_container/src
python analyze.py

# 分析結果確認
python analyze.py

# 分析結果確認
python commit_pre_processing.py

# 提出ファイル出力
python commit.py
