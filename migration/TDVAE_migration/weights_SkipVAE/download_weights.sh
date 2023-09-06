#!/bin/sh

desired_directory='migration/TDVAE_migration/weights_SkipVAE'

if [ -d "$desired_directory" ];
then
  cd "$desired_directory" || exit
fi

wget -c -O SkipVAE.zip 'https://zenodo.org/record/6583892/files/eval_SkipVAE40.zip?download=1'
unzip SkipVAE.zip