import  argparse
import  os
from utils.enmTypes import InputType,LossType,EvalType
from enum import  Enum

dir_prefix = os.getcwd()
print(f'current file path is {dir_prefix}')
def parse_arg():
    parser = argparse.ArgumentParser()

    # General Set
    parser.add_argument('--path', type = str, default=dir_prefix)
    parser.add_argument('--Device',type= str, default= 'cpu')
    parser.add_argument('--split_type',type = str,default='normal')
    parser.add_argument('--split_percent',type = list,default= [0.7,0.1,0.2])
    parser.add_argument('--compression',type= str,default='gzip')
    parser.add_argument('--random_seed',type= int,default= 1228)
    parser.add_argument('--negative_generate',type= bool,default=False)
    parser.add_argument('--binary_threshold',type= int ,default= 4)
    parser.add_argument('--plot_fig', type=bool, default=False )

    # graph simulated set
    parser.add_argument("--graph_type",type= str, default= 'barabasi-albert')
    parser.add_argument('--graph_degree',type= int,default= 3)
    parser.add_argument('--graph_sem_type',type= str,default= 'linear-gauss')
    parser.add_argument('--graph_linear_type',type= str, default= 'nonlinear_2')
    parser.add_argument('--data_sample_size', type=int, default=5000)
    parser.add_argument('--data_variable_size', type=int, default=5)
    parser.add_argument('--x_dims', type=int, default=1)


    # data Set
    parser.add_argument('--data_type', type= InputType,default= InputType.USERWISE)
    parser.add_argument('--eval_type', type= EvalType, default= EvalType.FULLSORT)
    parser.add_argument('--val_type', type= EvalType, default= EvalType.FULLSORT)
    parser.add_argument('--negative_sample_sort_nums',type= int,default= 99)
    parser.add_argument('--data_name',type= str,default='ml-100k')
    parser.add_argument('--sample',type= bool,default= False)
    parser.add_argument('--negative_number',type= int,default= 1)
    parser.add_argument('--mask',type= bool,default= False)
    parser.add_argument('--graph',type= bool,default= False)
    parser.add_argument('--batchsize',type= int,default= 2048)
    parser.add_argument('--num_worker', type= int, default= 2)
    parser.add_argument('--data_binary',type= bool,default= True)

    # Ranker Set
    parser.add_argument('--topk',type= list, default=[1,5,10,20,30,50,100])
    parser.add_argument('--metrics',type= list, default=['hit','mrr','map','recall','ndcg','precision','avg_pop','tail_percent'])
    parser.add_argument('--tail_ratio',type = float, default= 0.9)
    parser.add_argument('--use_price', type=bool, default=False)

    # evaluator
    parser.add_argument('--sortType',type = str, default= 'descending')
    parser.add_argument('--patience_max', type = int, default= 30)

    # Trainer Set
    parser.add_argument('--model_name',type= str,default='MF')
    parser.add_argument('--lr',type= float,default=1e-4)
    parser.add_argument('--weight-decay',type= float,default=1e-4)
    parser.add_argument('--loss_name',type= LossType,default= LossType.BCE)
    parser.add_argument('--explicit_loss',type= bool,default= False)
    parser.add_argument('--file_name',type =str,default='MF')
    parser.add_argument('--embedding_size', type= int, default= 16)
    parser.add_argument('--save_result',type = bool, default= True)
    parser.add_argument('--epoch',type = int,default= 300)
    parser.add_argument("--validate_metric",type = str,default='recall')
    parser.add_argument('--validate_k',type = int ,default= 30)
    parser.add_argument('--performance_rerun', type=bool, default=False)

    # model parameter set
    parser.add_argument('--model_parameter',type= Enum,default= Enum)
    parser.add_argument('--k_max_iter',type = int,default= 1e2)
    parser.add_argument('--h_tol',type = int,default= 1e-8)

    #optimizer criterion set
    parser.add_argument('--use_two_step', type=bool, default=False)

    # hypertune set
    parser.add_argument('--hypertune', type=bool, default= False)
    parser.add_argument('--hypername', type=list, default=['concepts', 'alpha'])
    parser.add_argument(
        '--hypervalue', type=list,
        default=[
            [1, 2, 4, 8],
            [0, 10, 50, 100]
        ]
    )
    parser.add_argument('--hyperdatasets', type=list, default=['tafeng', 'epinions', 'lastfm', 'ml-10m',  'yelp', 'ml-20m'])


    args = parser.parse_args()

    return  args


