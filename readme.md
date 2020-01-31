# 初期構築
docker build -t anaconda .
docker-compose up -d

# よく使うコマンド
docker-compose exec dev bash

python titanic.py

git add .
git commit -m "WIP"

git fetch origin master
git pull origin master

git push origin master

<!-- docker run -it \
-p 8888:8888 \
--rm \
--name anaconda_container \ 
--mount type=bind,src=`pwd`,dst=/workdir  \
anaconda

docker run -it -p 8888:8888 --name anaconda_container \ 
--mount type=bind,src=`pwd`,dst=/workdir anacond -->