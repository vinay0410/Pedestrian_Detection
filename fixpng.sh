find INRIAPerson/ -name "*.png" -type f > list.txt

while read p; do
  name=$(echo "$p" | cut -f 1 -d '.')
  pngfix --suffix="fixed" "$p"
  rm "$p"
done <list.txt

find INRIAPerson/ -name "*.pngfixed" -type f > list.txt

while read p; do
  name=$(echo "$p" | cut -f 1 -d '.')
  mv "$p" "$name.png"
done <list.txt
