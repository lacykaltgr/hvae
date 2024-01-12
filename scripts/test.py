from config import get_hparams
from hvae_backbone import testing

def main():
    hparams = get_hparams()
    testing(hparams)

if __name__ == '__main__':
    main()
