import torch
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())
def logit2mask(logit):
    b, c, h, w = logit.size()
    parsing = torch.argmax(logit, dim=1)
    parsing[parsing == 16] = 0
    parsing[parsing == 18] = 0
    # Binarization
    parsing[parsing > 0] = 1
    parsing = parsing.view(b, 1, h, w)
    return parsing.type(torch.cuda.FloatTensor)