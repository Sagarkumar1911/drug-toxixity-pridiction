import dgl
import torch

print('PyTorch :', torch.__version__)
print('DGL     :', dgl.__version__)

try:
    g = dgl.graph(([0, 1], [1, 2]))
    g = g.to('cuda')
    print('GPU test: PASSED')
except Exception as e:
    print('GPU test: FAILED —', e)