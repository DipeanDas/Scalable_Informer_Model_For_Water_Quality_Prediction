# main.py

import torch
from exp.exp_informer import Exp_Informer
from config import get_config

def main():
    args = get_config()

    print(f">>>>>>>>>> Start training: {args['model_id']} >>>>>>>>>>>>>>>")
    exp = Exp_Informer(args)
    exp.train()

    print(">>>>>>> Testing... <<<<<<<")
    exp.test()

if __name__ == '__main__':
    main()