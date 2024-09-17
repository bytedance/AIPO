#!/bin/bash

# Run debug.sh in sub-directories if exist
for DIR in $(ls -d configs/experiments/recipes/*/); do
  if [ -f "$DIR/debug.sh" ]; then
    echo ">>> Running debug.sh in $DIR"
    bash "$DIR/debug.sh"
  else
    echo ">>> No debug.sh in $DIR"
  fi
done
