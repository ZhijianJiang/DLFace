# ky2371
# This shell script will count how many classes have more than a certain amount of images in the dataset
mkdir "$2_classified"

classes=0

while IFS='' read -r line || [[ -n "$line" ]]; do
        if [ $(find $2 -name "*$line*" | wc -l) -gt $3 ]
        then
        	classes=`expr $classes + 1`
        fi
done < "$1"

echo "$classes classes have more than $3 images."