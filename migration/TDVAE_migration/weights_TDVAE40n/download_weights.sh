#!/bin/sh

desired_directory='migration/TDVAE_migration/weights_TDVAE40n'

if [ -d "$desired_directory" ];
then
  cd "$desired_directory" || exit
fi

wget -c -O TDVAE_40n.zip 'https://zenodo.org/record/6584667/files/eval_TDVAE40n.zip?download=1'
unzip TDVAE_40n.zip