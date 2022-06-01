from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, matthews_corrcoef
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import torch
except ImportError:
    torch = None

### Evaluator for graph classification
class Evaluator:
    def __init__(self, metric="acc", num_tasks=14):

        self.num_tasks = num_tasks
        self.eval_metric = metric


    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'ap' or self.eval_metric == 'rmse' or self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()


            ## check type
            if not isinstance(y_true, np.ndarray):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            return y_true, y_pred

        elif self.eval_metric == 'F1':
            if not 'seq_ref' in input_dict:
                raise RuntimeError('Missing key of seq_ref')
            if not 'seq_pred' in input_dict:
                raise RuntimeError('Missing key of seq_pred')

            seq_ref, seq_pred = input_dict['seq_ref'], input_dict['seq_pred']

            if not isinstance(seq_ref, list):
                raise RuntimeError('seq_ref must be of type list')

            if not isinstance(seq_pred, list):
                raise RuntimeError('seq_pred must be of type list')

            if len(seq_ref) != len(seq_pred):
                raise RuntimeError('Length of seq_true and seq_pred should be the same')

            return seq_ref, seq_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def eval(self, input_dict):

        results = {}

        y_true, y_pred = self._parse_and_check_input(input_dict)

        results["mcc"] =  matthews_corrcoef(y_true=y_true, y_pred=y_pred)

        results["acc"] = accuracy_score(y_true= y_true, y_pred=y_pred)

        results["f1"] = f1_score(y_true= y_true, y_pred=y_pred, average='weighted')

        return results


    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc' or self.eval_metric == 'ap':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_graph, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_graph, num_task)\n'
            desc += 'where y_pred stores score values (for computing AUC score),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
            desc += 'nan values in y_true are ignored during evaluation.\n'
        elif self.eval_metric == 'rmse':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_graph, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_graph, num_task)\n'
            desc += 'where num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
            desc += 'nan values in y_true are ignored during evaluation.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
        elif self.eval_metric == 'F1':
            desc += '{\'seq_ref\': seq_ref, \'seq_pred\': seq_pred}\n'
            desc += '- seq_ref: a list of lists of strings\n'
            desc += '- seq_pred: a list of lists of strings\n'
            desc += 'where seq_ref stores the reference sequences of sub-tokens, and\n'
            desc += 'seq_pred stores the predicted sequences of sub-tokens.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'ap':
            desc += '{\'ap\': ap}\n'
            desc += '- ap (float): Average Precision (AP) score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'rmse':
            desc += '{\'rmse\': rmse}\n'
            desc += '- rmse (float): root mean squared error averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'F1':
            desc += '{\'F1\': F1}\n'
            desc += '- F1 (float): F1 score averaged over samples.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc


if __name__ == '__main__':
    evaluator = Evaluator('ogbg-code2')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    seq_ref = [['tom', 'is'], ['he'], ['he'], ['hey', 'fea', 'he'], ['alpha'], ['fe4qfq', 'beta'], ['aa']]
    seq_pred = [['tom', 'is'], ['he'], ['he'], ['hey', 'he', 'fea'], ['alpha'], ['beta', 'fe4qfq'], ['aa']] # [['tom', 'is'] , ['he'], ['the', 'he'], ['hey', 'fea', 'he'], ['alpha'], ['beta', 'fe4qfq', 'c', 'fe4qf'], ['']]
    input_dict = {'seq_ref': seq_ref, 'seq_pred': seq_pred}
    result = evaluator.eval(input_dict)
    print(result)

    # exit(-1)

    evaluator = Evaluator('ogbg-molpcba')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = torch.tensor(np.random.randint(2, size = (100,128)))
    y_pred = torch.tensor(np.random.randn(100,128))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    evaluator = Evaluator('ogbg-molhiv')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = torch.tensor(np.random.randint(2, size = (100,1)))
    y_pred = torch.tensor(np.random.randn(100,1))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    ### rmse case
    evaluator = Evaluator('ogbg-mollipo')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randn(100,1)
    y_pred = np.random.randn(100,1)
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    ### acc
    evaluator = Evaluator('ogbg-ppa')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randint(5, size = (100,1))
    y_pred = np.random.randint(5, size = (100,1))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)


