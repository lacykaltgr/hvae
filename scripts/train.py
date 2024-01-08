from config import get_hparams
from hvae_backbone import training

def main():
    hparams = get_hparams()
    training(hparams)

if __name__ == '__main__':
    main()
