# Get hauWE Embeddings
home_dir=$(pwd)

mkdir embeddings
cd embeddings

mkdir hausa
cd hausa

wget https://zenodo.org/record/6997888/files/hauWE-Models.zip
unzip hauWE-Models.zip
rm hauWE-Models.zip

cd $home_dir

# Get Panlex-Lexicon-Extractor and PanLex Lite

# make directory Panlex
mkdir Panlex
cd Panlex

# clone or pull Panlex-Lexicon-Extractor
repo=Panlex-Lexicon-Extractor
url=https://github.com/abumafrim/Panlex-Lexicon-Extractor.git
if cd $repo; then git pull; else git clone $url $repo; fi

# make data directory
mkdir -p data
cd data

# download and prepare PanLex Lite
wget https://db.panlex.org/panlex_lite.zip
unzip panlex_lite.zip
rm panlex_lite.zip

cd $home_dir

# clone or pull Panlex-Lexicon-Extractor
repo=vecmap
url=https://github.com/artetxem/vecmap.git
if cd $repo; then git pull; else git clone $url $repo; fi

cd $home_dir