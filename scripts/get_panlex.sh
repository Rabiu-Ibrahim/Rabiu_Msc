home_dir=$(pwd)

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