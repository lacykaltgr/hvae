#!/bin/sh

desired_directory='migration/TDVAE_migration/weights_TDVAE40'

if [ -d "$desired_directory" ];
then
  cd "$desired_directory" || exit
fi

wget -c -O TDVAE_40.zip 'https://zenodo.org/record/6584407/files/eval_TDVAE40.zip?download=1'
unzip TDVAE_40.zip