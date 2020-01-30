docker build -t anaconda .

docker-compose up -d

<!-- docker run -it \
-p 8888:8888 \
--rm \
--name anaconda_container \ 
--mount type=bind,src=`pwd`,dst=/workdir  \
anaconda

docker run -it -p 8888:8888 --name anaconda_container \ 
--mount type=bind,src=`pwd`,dst=/workdir anacond -->