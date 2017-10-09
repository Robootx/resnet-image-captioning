# resnet-image-captioning
show attention and tell with resnet  
Adapt from [show attend and tell](https://github.com/yunjey/show-attend-and-tell), replace VGG to resnet, and can be trained together with LSTM. resnet code borrowed and adapted from [models/official/resnet/](https://github.com/tensorflow/models/tree/master/official/resnet)  
Use the output of block_layer3 in resnet50 as image features get result in   
val set:  
> Bleu_1: 0.660386  
> Bleu_2: 0.447982  
> Bleu_3: 0.305375  
> Bleu_4: 0.213699  
> METEOR: 0.213692  
> ROUGE_L: 0.515579  
> CIDEr: 0.665676

test set:  
> Bleu_1: 0.623284  
> Bleu_2: 0.399399  
> Bleu_3: 0.260130  
> Bleu_4: 0.174152  
> METEOR: 0.191364  
> ROUGE_L: 0.486628  
> CIDEr: 0.501525  
