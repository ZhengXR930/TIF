#!/bin/bash
# set -x

for f in $(cat ./list_of_apps_1)
do
	dir='./apps_folder/apk/'
        tmpdir='tmp_1/'
  	cp $dir$f $dir$tmpdir$f
        python wrapper.py $dir$tmpdir$f $dir$tmpdir
  	rm -f -r "$dir$tmpdir"smali 
  	rm -f -r "$dir$tmpdir"unpack
  	mv "$dir$tmpdir"static.log "$dir$f.log"
  	mv "$dir$tmpdir"static.json "$dir$f.json"
done

