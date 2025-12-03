# training arguments

import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument('--image_height', type=int, default=224,
                        help="image_height")
    parser.add_argument('--image_width', type=int, default=224,
                        help="image_width")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=12,
                        help="number of global rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--val_global_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=1,
                        help="local batch size: B")

    # other arguments
    parser.add_argument('--num_classes', type=int, default=5, help="number \
                        of classes")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
