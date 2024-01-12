from config import get_hparams
from hvae_backbone import migrate

def main():
    hparams = get_hparams()
    migrate(hparams)

if __name__ == '__main__':
    main()
