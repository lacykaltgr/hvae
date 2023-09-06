#!/bin/sh

desired_directory='migration/TDVAE_migration/weights_TDVAE40125'

if [ -d "$desired_directory" ];
then
  cd "$desired_directory" || exit
fi

wget -c -O TDVAE_40125.zip 'https://zenodo.org/record/6584838/files/eval_TDVAE40125.zip?download=1'
unzip TDVAE_40125.zip