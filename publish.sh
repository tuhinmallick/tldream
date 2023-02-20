#!/usr/bin/env bash
set -e

pushd ./tldream-frontend
yarn run build
cd ./apps/www
yarn run export
popd

rm -r ./tldream/out
cp -r ./tldream-frontend/apps/www/out ./tldream/out

rm -r -f dist build
python3 setup.py sdist bdist_wheel
twine upload dist/*
