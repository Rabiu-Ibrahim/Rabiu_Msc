home_dir=$(pwd)

mkdir embeddings
cd embeddings

mkdir hausa
cd hausa

wget https://zenodo.org/record/6997888/files/hauWE-Models.zip
unzip hauWE-Models.zip
rm hauWE-Models.zip

cd $home_dir