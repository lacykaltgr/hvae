#!/bin/sh

desired_directory='migration/TDVAE_migration/weights_LinearVAE'

if [ -d "$desired_directory" ];
then
  cd "$desired_directory" || exit
fi

wget -c -O LinearVAE.zip 'https://zenodo.org/record/6582573/files/eval_LinearVAE40.zip?download=1'
unzip LinearVAE.zip