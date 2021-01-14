rm -r ./docs/rst
rm -r ./docs/html
cp -R ./docs/backup ./docs/rst
sphinx-apidoc -o ./docs/rst ./
sphinx-build -b html ./docs/rst  ./docs/html
