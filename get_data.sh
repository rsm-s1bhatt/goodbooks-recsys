#!/usr/bin/env bash
# Download goodbooks-10k (Zajac, 2017) into data/raw/
set -e
mkdir -p data/raw
for f in ratings.csv books.csv to_read.csv book_tags.csv tags.csv; do
  curl -sSL -o data/raw/$f https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/$f
done
