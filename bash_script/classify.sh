# ky2371
# This shell script will classify faces according to their annotations
mkdir "$2_classified"

while IFS='' read -r line || [[ -n "$line" ]]; do
        if [ $(find $2 -name "*$line*" | wc -l) -gt $3 ]
        then
            mkdir $2_classified/$line
            cp $2/*$line* $2_classified/$line
        fi
done < "$1"