set -e

fileCount=`ls tmp_data | wc -l`
if [ $fileCount -le 1 ]; then
  echo "gpfs mount pack_data failed！"
  exit 1
fi

fileCount=`ls tmp_orig_data | wc -l`
if [ $fileCount -le 1 ]; then
  echo "gpfs mount orig_data failed！"
  exit 1
fi

echo "gpfs mount success!"
