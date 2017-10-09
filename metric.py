from core.bleu import evaluate                   
from core.utils import *

def main():
    scores = evaluate(data_path='./data', split='test', get_scores=True)
    write_bleu(scores=scores, path='model/resnet_50/block_layer4', epoch=0)

if __name__ == '__main__':
    main()