from itertools import product, combinations
from collections import defaultdict

# Training parameters:
# model, mini_batch_size, criterion, optimizer, nb_epochs,
# weight_loss_classes, weight_loss_pairs,
# freeze_epochs,
# convert_classes_to_one_hot, convert_ioe_to_one_hot

# Model parameters:
# Siamese: conv_block_parameters, fc_parameters, predict
# CNN: conv_block_parameters, fc_parameters, predict
# FC: fc_parameters, predict
# ConvBlock:       ch1=64, ch2=64,
#                  conv1_kernel=3, conv2_kernel=2,
#                  use_max_pool1=True, max_pool1_kernel=3, max_pool_stride1=1,
#                  use_max_pool2=True, max_pool2_kernel=2, max_pool_stride2=1,
#                  dropout1=0.0, dropout2=0.0,
#                  activation1=nn.ReLU(), activation2=nn.ReLU(),
#                  use_batch_norm=True, use_skip_connections
# FCBlock:         fc, out=10,
#                  dropout=0.0,
#                  activation1=nn.ReLU(), activation2=nn.ReLU(),
#                  use_batch_norm=True


