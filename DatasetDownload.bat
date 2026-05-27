@echo off
Title BanglaLekha-Isolated

mkdir Dataset .
cd Dataset

curl -L -o BanglaLekha-Isolated.zip https://data.mendeley.com/public-files/datasets/hf6sf8zrkc/files/ffe73a3a-5999-4ad2-970f-3e14c141cdb1/file_downloaded
unzip BanglaLekha-Isolated.zip

del BanglaLekha-Isolated.zip

exit