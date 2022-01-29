#!/bin/bash
rm -f ./asset/time-*

for ((thread = 1; thread <= 8; thread++)) do
  for ((size = 512; size <= 2048; size += 512)) do
    echo ------------------------------------------
    echo Size = $size
		echo Thread = $thread
		./bin/test $size $size $size $thread
    echo
	done
done