from config import get_hparams
from hvae_backbone import analysis

def main():
    hparams = get_hparams()
    analysis(hparams)

if __name__ == '__main__':
    main()
