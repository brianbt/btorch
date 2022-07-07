import math
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

import btorch

class GridSearchCV:
    """Exhaustive search over specified parameter values for an estimator.
    
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        
    GridSearchCV requires a btorch.nn.Module as model. In particular, the ".fit" and "evaluate" must be implemented.
    
    GridSearchCV search over parameters to find the best combination of parameters that yields the **lowest loss**.
    Here loss is smaller better. Only the best model is saved.
    
        Args:
            model (nn.Module): 
              A btorch model.
            param_grid (dict): 
              Dictionary with parameters names (str) as keys and lists of parameter settings to try as values. 
              The parameters are those in ``NET(*args)``.  
            optim_param_grid (dict, optional): 
              Dictionary with parameters names (str) as keys and lists of parameter settings to try as values. 
              The parameters are those in ``torch.optim.OPTIM(*args)``.
              Defaults to None.
            lossfn_param_grid (dict, optional):
              Dictionary with parameters names (str) as keys and lists of parameter settings to try as values. 
              The parameters are those in ``torch.nn.LOSS(*args)``.
              Defaults to None.
            lr_s_param_grid (dict, optional):
              Dictionary with parameters names (str) as keys and lists of parameter settings to try as values. 
              The parameters are those in ``torch.optim.lr_scheduler.LRS(*args)``.
              Defaults to None.
            
                Note:
                  For ``param_grid``, ``optim_param_grid``, ``lossfn_param_grid``, and ``lr_s_param_grid``,
                  the keys names should NOT start with [``optim_``, ``lossfn_``, ``lr_s_``]. They are resevered.
            
            scoring (Callable, optional): A scoring function that take in ``y_true`` and ``model_output``.
              Usually, this is your evaluation metric, like accuracy.
              If provided, this method return a dict that include both loss and score.
              This scoring function should return the **sum** (set ``reduction=sum``) of the score of a batch.
              The function signature must be ``scoring(y_true=, model_output=)``.
            cv (int, optional): 
              Determines the number of fold to split. Defaults to 3.
            
            _config (dict): config for btorch model.
            _lossfn (callable): loss function for btorch model [functools.partial Class Constructor].
            _optimizer (callable): optimizer for btorch model [functools.partial Class Constructor].
            _lr_scheduler (callable): lr_scheduler for btorch model [functools.partial Class Constructor].
            
                Note:
                  For ``_lossfn``, ``_optimizer``, and ``_lr_scheduler``, you must wrap the constructor using ``functools.partial()``.
                  You should define all non-searching parameters in ``functools.partial()``.
                  EG, If you want grid search ``lr`` in ``_optimizer``, you should not put ``lr`` in ``functools.partial()``.   
                
        Attributes: 
            cv_results_ (dict): 
              A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
            best_model_ (nn.Module): 
              The best model.
            best_loss_ (float): 
              The best loss.
            best_score_ (float): 
              The best score if ``scoring`` is provided.
            best_params_ (dict): 
              The best parameters.
            
        Examples:
            >>> param_grid = {'hidden_dim':[20,30,40]}
            >>> optim_grid = {'lr':[0.01, 0.1]}
            >>> a = GridSearchCV(Net, param_grid, optim_param_grid=optim_grid, scoring=accuarcy)

            >>> # Define the lossfn, optimizer, those thing as usual.
            >>> # Something different is that you are now passing the Class to them, instead of Class_instance
            >>> # For optimizer and lr_scheduler, you must use ``partial`` to wrap it first
            >>> # Since we would like to search through the learning_rate, you leave the ``lr`` arg empty in ``partial``
            >>> from functools import partial
            >>> a._lossfn = nn.nn.BCEWithLogitsLoss
            >>> # a._lossfn = nn.nn.BCEWithLogitsLoss() # WRONG
            >>> a._optimizer = partial(torch.optim.Adam, betas=(0.9, 0.999))
            >>> # a._optimizer = torch.optim.Adam # WRONG
            >>> a._lr_scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=2)
            >>> # a._lr_scheduler = torch.optim.lr_scheduler.StepLR # WRONG
            >>>  a._config['max_epoch'] = 2
            
            >>> a.fit(x)
        """

    def __init__(self, model, param_grid, optim_param_grid=None, lossfn_param_grid=None,
                 lr_s_param_grid=None, scoring=None, cv=3, **kwargs):
        self.model = model
        self.total_param_grid = dict()
        self._param_grid = None
        self._optim_param_grid = None
        self._lossfn_param_grid = None
        self._lr_s_param_grid = None
        self.param_grid = param_grid
        self.optim_param_grid = optim_param_grid
        self.lossfn_param_grid = lossfn_param_grid
        self.lr_s_param_grid = lr_s_param_grid
        self.scoring = scoring

        self.cv = cv
        self.cv_results_ = {'params': []}
        self.cv_results_lookup = {'mean_train_loss': [], 'mean_test_loss': [], 'mean_train_score': [],
                                  'mean_test_score': []}
        for i in range(cv):
            self.cv_results_[f'split{i}_train_loss'] = []
            self.cv_results_[f'split{i}_test_loss'] = []
            self.cv_results_lookup['mean_train_loss'].append(f'split{i}_train_loss')
            self.cv_results_lookup['mean_test_loss'].append(f'split{i}_test_loss')
            if self.scoring is not None:
                self.cv_results_[f'split{i}_train_score'] = []
                self.cv_results_[f'split{i}_test_score'] = []
                self.cv_results_lookup['mean_train_score'].append(f'split{i}_train_score')
                self.cv_results_lookup['mean_test_score'].append(f'split{i}_test_score')

        self.best_model_ = None
        self.best_loss_ = math.inf
        self.best_score_ = None
        self.best_params_ = None

        self._config = dict()
        self._lossfn = None
        self._optimizer = None
        self._lr_scheduler = None

    def __call__(self, *args, **kwargs):
        if self.best_model_ is None:
            raise ValueError("You must call .fit() first")
        return self.best_model_(*args, **kwargs)

    @property
    def param_grid(self):
        return self._param_grid

    @param_grid.setter
    def param_grid(self, d):
        if d is not None:
            self._param_grid = d
            self.total_param_grid.update(d)

    @property
    def lr_s_param_grid(self):
        return self._lr_s_param_grid

    @lr_s_param_grid.setter
    def lr_s_param_grid(self, d):
        if d is not None:
            new_d = {}
            for k in d.keys():
                new_d[f"lr_s_{k}"] = d[k]
            self._lr_s_param_grid = new_d
            self.total_param_grid.update(new_d)

    @property
    def lossfn_param_grid(self):
        return self._lossfn_param_grid

    @lossfn_param_grid.setter
    def lossfn_param_grid(self, d):
        if d is not None:
            new_d = {}
            for k in d.keys():
                new_d[f"lossfn_{k}"] = d[k]
            self._lossfn_param_grid = new_d
            self.total_param_grid.update(new_d)

    @property
    def optim_param_grid(self):
        return self._optim_param_grid

    @optim_param_grid.setter
    def optim_param_grid(self, d):
        if d is not None:
            new_d = {}
            for k in d.keys():
                new_d[f"optim_{k}"] = d[k]
            self._optim_param_grid = new_d
            self.total_param_grid.update(new_d)

    def all_combination_in_dict_of_list(self, dict_of_list):
        """ Get all combination from a dict of list.
        https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
        """
        import itertools
        keys, values = zip(*dict_of_list.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts

    def extract_single_config(self, x):
        """Input a curr_params_dict, output the corresponding component params.
        """
        model_params = {}
        lossfn_params = {}
        optim_params = {}
        lr_s_params = {}
        for k, v in x.items():
            if k.startswith('lossfn_'):
                lossfn_params[k[7:]] = v
            elif k.startswith('optim_'):
                optim_params[k[6:]] = v
            elif k.startswith('lr_s_'):
                lr_s_params[k[5:]] = v
            else:
                model_params[k] = v
        return {'model_params': model_params, 'lossfn_params': lossfn_params,
                'optim_params': optim_params, 'lr_s_params': lr_s_params}

    def init_model(self, curr_config, *args, **kwargs):
        curr_params = self.extract_single_config(curr_config)
        model = self.model(**curr_params['model_params'])
        model._lossfn = self._lossfn(**curr_params['lossfn_params'])
        model._optimizer = self._optimizer(model.parameters(), **curr_params['optim_params'])
        model._lr_scheduler = self._lr_scheduler(model._optimizer, **curr_params['lr_s_params'])
        if self._config is not None:
            model._config = self._config
        return model

    def score(self, net, x):
        """run net.evaluate and wrap the output into a dict and return it.
        """
        results = net.evaluate(x, scoring=self.scoring)
        if isinstance(results, dict):
            return results
        else:
            return {'loss': results}

    def fit(self, x=None, y=None, **kwargs):
        """Run fit with all sets of parameters. This will call .fit() and .evaluate().
        
        Args:
            x: Input data. It could be
                - torch.tensor in batch node, starting with (N, *)
                - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, targets)``
              
              Note:
                ``torch.utils.data.DataLoader`` is NOT supported.
                
            y: Target data. Like the input data ``x``,
              it should be torch.Tensor.
              If ``x`` is a dataset, ``y`` should
              not be specified (since targets will be obtained from ``x``).
              
            **kwargs: Other arguments for btorch model ``.fit()``.
        """
        if x is None:
            raise ValueError("x is not provided")
        # Make sure x, y is in Dataset format
        if isinstance(x, torch.Tensor) or isinstance(x, (tuple, list)):
            assert y is not None, f"x is {type(x)}, expected y to be torch.Tensor or List[Tensor]"
            x = btorch.utils.tensor_to_Dataset(x, y)
        assert isinstance(x, torch.utils.data.Dataset), 'x cannot be DataLoader'
        # Split into ``cv`` folds
        split_num = [len(x) // self.cv for _ in range(self.cv - 1)]
        split_num.append(len(x) - sum(split_num))
        split_data = torch.utils.data.random_split(x, split_num)

        for curr_params in self.all_combination_in_dict_of_list(self.total_param_grid):
            self.cv_results_['params'].append(str(curr_params))
            for curr_split in range(self.cv):
                curr_x_train = ConcatDataset([da for i, da in enumerate(split_data) if i != curr_split])
                curr_x_eval = split_data[curr_split]
                curr_model = self.init_model(curr_params)
                curr_model.fit(curr_x_train, **kwargs)
                train_results = self.score(curr_model, curr_x_train)
                test_results = self.score(curr_model, curr_x_eval)
                self.cv_results_[f'split{curr_split}_train_loss'].append(train_results['loss'])
                self.cv_results_[f'split{curr_split}_test_loss'].append(test_results['loss'])
                if self.scoring is not None:
                    self.cv_results_[f'split{curr_split}_train_score'].append(train_results['score'])
                    self.cv_results_[f'split{curr_split}_test_score'].append(test_results['score'])
                if test_results['loss'] < self.best_loss_:
                    self.best_model_ = curr_model
                    self.best_loss_ = test_results['loss']
                    if self.scoring is not None:
                        self.best_score_ = test_results['score']
                    self.best_params_ = curr_params
        # Calc mean, std, rank
        df = pd.DataFrame(self.cv_results_)
        df['mean_train_loss'] = df[self.cv_results_lookup['mean_train_loss']].mean(1)
        df['mean_test_loss'] = df[self.cv_results_lookup['mean_test_loss']].mean(1)
        df['std_train_loss'] = df[self.cv_results_lookup['mean_train_loss']].mean(1)
        df['std_test_loss'] = df[self.cv_results_lookup['mean_test_loss']].mean(1)
        df['rank_train_loss'] = df['mean_train_loss'].rank()
        df['rank_test_loss'] = df['mean_test_loss'].rank()
        if self.scoring is not None:
            df['mean_train_score'] = df[self.cv_results_lookup['mean_train_score']].mean(1)
            df['mean_test_score'] = df[self.cv_results_lookup['mean_test_score']].mean(1)
            df['std_train_score'] = df[self.cv_results_lookup['mean_train_score']].mean(1)
            df['std_test_score'] = df[self.cv_results_lookup['mean_test_score']].mean(1)
            df['rank_train_score'] = df['mean_train_score'].rank()
            df['rank_test_score'] = df['mean_test_score'].rank()
        self.cv_results_ = df.to_dict()
