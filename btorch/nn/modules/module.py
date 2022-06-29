import warnings
from tqdm import tqdm
from torchinfo import summary
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import btorch
from btorch.utils.load_save import save_model, resume

class Module(nn.Module):
    """Base class for all neural network modules.

    Your models should also subclass this class.
    btorch.nn.Module is inhernet from pytorch.nn, hence, all syntax is same as pytorch
    You can replace your `from torch import nn` as `from btorch import nn`
    This class provides some highlevel training method like Keras:
        - .fit()
        - .evaluate()
        - .predict()
        - .overfit_small_batch()
    These highlevel method will use the core class methods which should be overrided for advanced use.
    The core class methods are:
        - .train_net()
        - .train_epoch()
        - .test_epoch()
        - .predict_()
        - .overfit_small_batch_()
    All of above classmethods can be overrided at your need. 
    Notice:
        When overriding instance method, call classmethod via `self.`
        When overriding class method, remember to put `@classmethod`.
        If you want to use core class method directly in real use case, use as follow:
        >>> class Model(nn.Module):
        >>>     ...
        >>> mod = Model()
        >>> mod.train_net(...)   # correct
        >>> Model.train_net(...) # wrong
        When overriding class method and if you want to use instance method (eg. fit),
        you should keep the exact SAME signature in the class method. 
        Inside class method:
            -> call instance variable via `net.`
            -> call instance method via `net.`
            -> call class method via `cls.`

    Hierarchy View:
    .fit
    └── @train_net -> {train_loss, test_loss}
        ├── @train_epoch -> train_loss
        └── @test_epoch -> test_loss

    .evaluate -> test_loss
    └── @test_epoch -> test_loss

    .predict -> prediction
    └── @predict_ -> prediction

    .overfit_small_batch
    └── @overfit_small_batch_
        └── @train_epoch -> train_loss

    If you decided to use the highlevel training loop. Please set 
        - self._lossfn (default:pytorch Loss Func)[required]
        - self._optimizer (default:pytorch Optimizer)[required]
        - self._lr_scheduler (default:pytorch lr_Scheduler)[optional]
        - self._config (default:dict)[optional]. Contains all setting and hyper-parameters for training loops
          For Defaults usage, it accepts:
            start_epoch: start_epoch idx
            max_epoch: max number of epoch
            device: either `cuda` or `cpu` or `auto`
            save: save model path
            resume: resume model path. Override start_epoch
            val_freq = freq of running validation
        - self._history (default:dict)[optional]. All loss, evaluation metric should be here.
    You can set them to be a pytorch instance (or a dictionary, for advanced uses)
    The default guideline is only for the default highlevel functions.

    Other high level utils methods are:
        - .set_gpu()
        - .set_cpu()
        - .auto_gpu()
        - .save()
        - .load()
        - .summary()

    All the classmethods in this class can be taken out and use them alone. 
    They are a good starter code for traditional PyTorch training code.
    """
    def __init__(self) -> None:
        super(Module, self).__init__()
        self._lossfn = None
        self._optimizer = None
        self._lr_scheduler = None
        self._config = {
                        "start_epoch": 0,
                        "max_epoch": 10,
                        "device": "cpu",
                        "save": None,
                        "resume": None,
                        "val_freq": 1
                        }
        self._history = None

    def fit(self, x=None, y=None, batch_size=8, epochs=None, shuffle=True, drop_last=False,
            validation_split=0.0, validation_data=None, validation_batch_size=8, validation_freq=1,
            initial_epoch=None, workers=1):
        """Trains the model for a fixed number of epochs (iterations on a dataset).
        
        Keras like fit method. All arguments follows Keras usage.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        It uses .train_net()

        Args:
            X: Input data. It could be:
              - torch.tensor in batch node, starting with (N, *)
              - a `torch.utils.data.Dataset` dataset. Should return a tuple of `(inputs, targets)`
              - a `torch.utils.data.Dataloader`. All other dataset related argument will be ignored, if provided.
            y: Target data. Like the input data `x`,
              it should be torch.tensor.
              If `x` is a dataset, generator or dataloader, `y` should
              not be specified (since targets will be obtained from `x`).
            batch_size (int, optional): Defaults to 10.
            epochs (int, optional): Defaults to 1.
            shuffle (bool, optional): Defaults to True.
            drop_last (bool, optional): Shuffle the data or not. 
            validation_split (optional): Float between 0 and 1.
              Fraction of the training data to be used as validation data.
              The model will set apart this fraction of the training data,
              will not train on it. This argument is
              not supported when `x` is a Dataset or Dataloader.
              Defaults to 0.
            validation_data (optional): Data on which to evaluate the loss 
              and any model metrics at the end of each epoch. 
              The model will not be trained on this data. Defaults to None.
              `validation_data` will override `validation_split`. 
              `validation_data` could be:
                - tuple of torch.tensor, tuple(X, y)
                - a `torch.utils.data.Dataset` dataset. Should return a tuple of `(inputs, targets)`
                - a `torch.utils.data.Dataloader`. All other dataset related argument will be ignored.
            validation_batch_size (optional): batch size of validation data
            validation_freq (optional): runs validation every x epochs.
            initial_epoch (optional): start epoch. Return from `btorch.utils.load_save.resume`
            workers (optional): num_workers for dataloader
        """
        if x is None:
            raise ValueError("x is not provided")
        if self._lossfn is None or self._optimizer is None:
            raise ValueError("`self._lossfn` and `self._optimizer` is not set.")

        self._config['max_epoch'] = epochs if epochs is not None else self._config['max_epoch']
        self._config['start_epoch'] = initial_epoch if initial_epoch is not None else 0
        self._config['val_freq'] = validation_freq

        pin_memory = True if self._config.get('device', 'cpu')=='cuda' else False
        x_eval = None
        eval_loader = None

        # Pre-process train_loader
        if isinstance(x, torch.utils.data.DataLoader):
            train_loader = x
        else:
            if isinstance(x, torch.Tensor):
                if y is None:
                    raise ValueError(f"x is {type(x)}, expected y to be torch.Tensor")
                if validation_split != 0:
                    x, x_eval, y, y_eval = train_test_split(x, y, test_size=validation_split)
                    x_eval = TensorDataset(x_eval, y_eval)
                x = TensorDataset(x,y)
            elif isinstance(x, (tuple, list)):
                if isinstance(y, (tuple, list)):
                    if validation_split != 0:
                        splited = train_test_split(*x, *y, test_size=validation_split)
                        x = []
                        x_eval = []
                        for i in range(len(splited)):
                            if i%0==0:
                                x.append(splited[i])
                            else:
                                x_eval.append(splited[i])
                        x = TensorDataset(*x)
                        x_eval = TensorDataset(x_eval)
                    else:
                        x = TensorDataset(*x, *y)
                else:
                    if validation_split != 0:
                        splited = train_test_split(*x, y, test_size=validation_split)
                        x = []
                        x_eval = []
                        for i in range(len(splited)):
                            if i%0==0:
                                x.append(splited[i])
                            else:
                                x_eval.append(splited[i])
                        x = TensorDataset(*x)
                        x_eval = TensorDataset(*x_eval)
                    else:
                        x = TensorDataset(*x, y)
            train_loader = DataLoader(x, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                      pin_memory=pin_memory, drop_last=drop_last)
        # Pre-process eval_loader
        if validation_data is not None:
            if isinstance(validation_data, (tuple, list)):
                x_eval = TensorDataset(*validation_data)
            elif isinstance(validation_data, torch.utils.data.Dataset):
                x_eval = validation_data
            elif isinstance(validation_data, torch.utils.data.DataLoader):
                eval_loader = validation_data
                x_eval = None
            else:
                raise ValueError(f"validation_data doesn't support {type(validation_data)}")
        if x_eval is not None:
            eval_loader = DataLoader(x_eval, batch_size=validation_batch_size, num_workers=workers,
                                     pin_memory=pin_memory, drop_last=drop_last)


        self._history = self.train_net(net=self, criterion=self._lossfn, optimizer=self._optimizer,
                        trainloader=train_loader, testloader=eval_loader, lr_scheduler=self._lr_scheduler, config=self._config)

    def evaluate(self, x=None, y=None, batch_size=None, drop_last=False, workers=1, **kwargs):
        """Returns the loss value & metrics values for the model in test mode.

        Keras like evaluate method. All arguments follows Keras usage.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
        It uses .test_epoch()

        Args:
            X: Input data. It could be:
              - torch.tensor in batch node, starting with (N, *)
              - a `torch.utils.data.Dataset` dataset. Should return a tuple of `(inputs, targets)`
              - a `torch.utils.data.Dataloader`. All other dataset related argument will be ignored, if provided.
            y: Target data. Like the input data `x`,
              it should be torch.tensor.
              If `x` is a dataset, generator or dataloader, `y` should
              not be specified (since targets will be obtained from `x`).
            batch_size (int, optional): Defaults to 10.
            drop_last (bool, optional): Shuffle the data or not. 
            workers (optional): num_workers for dataloader
            """
        # Pre-process train_loader
        pin_memory = True if self._config.get('device', 'cpu')=='cuda' else False
        if isinstance(x, torch.utils.data.DataLoader):
            test_loader = x
        else:
            if isinstance(x, torch.Tensor):
                if y is None:
                    raise ValueError(f"x is {type(x)}, expected y to be torch.Tensor")
                x = TensorDataset(x,y)
            elif isinstance(x, (tuple, list)):
                if isinstance(y, (tuple, list)):
                    x = TensorDataset(*x, *y)
                else:
                    x = TensorDataset(*x, y)
            test_loader = DataLoader(x, batch_size=batch_size, num_workers=workers,
                                        pin_memory=pin_memory, drop_last=drop_last)
        return self.test_epoch(self, self._lossfn, test_loader, self._config.get("device", "cpu"))

    def predict(self, x, batch_size=8, return_combined=False):
        """Generates output predictions for input samples.

        Keras like predict method. All arguments follows Keras usage.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        It uses .predict_()

        TODO: handle when the .predict_() return dict

        Args:
            X: Input data. It could be:
              - torch.tensor in batch node, starting with (N, *)
              - a `torch.utils.data.Dataset` dataset. Should return a tuple of `(inputs, _)`
              - a `torch.utils.data.Dataloader`. All other dataset related argument will be ignored, if provided.
            batch_size (int, optional). Defaults to 8.
            return_combined (bool, optional). 
                if return from `self.predict_` is a list. Combine them into a single object.
                Apply torch.cat() on the output from .predict_() if return is list of tensor.
                Defaults to False.

        Returns:
            List[Tensor] or Tensor if return_combined 
        """
        if isinstance(x, torch.Tensor):
            _y = torch.zeros(x.shape[0])
            dataset = TensorDataset(x, _y)
        elif isinstance(x, torch.utils.data.Dataset):
            dataset = x
        if isinstance(x, torch.utils.data.DataLoader):
            loader = x
        else:
            loader = DataLoader(dataset, batch_size)
        out = self.predict_(self, loader, device=self._config.get('device', 'cpu'))
        if return_combined:
            if isinstance(out, list):
                if isinstance(out[0], dict):
                    tmp = {}
                    for dd in out:
                        for item in dd.keys():
                            if item not in tmp:
                                tmp[item] = []
                            tmp[item].append(dd[item])
                    return tmp
                elif isinstance(out[0], torch.Tensor):
                    return torch.cat(out)
                else:
                    raise NotImplementedError(f"Please manually handle the output type from `self.predict_`, got {type(out[0])}. You can also turn off `return_combined`.")
            else:
                warnings.warn(f"The output type from `self.predict_` is {type(out)}, return_combined only useful when it is `list`")
                return out
        else:
            return out
        

    def overfit_small_batch(self, x):
        self.overfit_small_batch_(self, self._lossfn, x, self._optimizer, self._config)

    @classmethod
    def train_net(cls, net, criterion, optimizer, trainloader, testloader=None, lr_scheduler=None, config=None):
        """Standard PyTorch training loop. Override this function when necessary.
        It uses .train_epoch() and .test_epoch()
        
        Args:
            net (nn.Module): this is equivalent to `self` or `forward`
            criterion (any): can be a loss function or a list/dict of loss functions. It will be used in `@train_epoch`
            optimizer ([type]): can be a optimizer or a list/dict of optimizers. It will be used in `@train_epoch`
            trainloader (torch.utils.data.Dataloader): can be a dataloader or a list/dict of dataloaders. It will be used in `@train_epoch`
            testloader (torch.utils.data.Dataloader, optional): can be a dataloader or a list/dict of dataloaders. It will be used in `@train_epoch`. Defaults to None.
            lr_scheduler (torch.optim.lr_scheduler, optional): Defaults to None.
            config (dict, optional): Config for training. Defaults to None.
        """
        if config is None:
            config = dict()
        start_epoch = config.get("start_epoch", 0)
        max_epoch = config.get("max_epoch", 10)
        device = config.get("device", "cpu")
        save_path = config.get("save", None)
        resume_path = config.get("resume", None)
        val_freq = config.get("val_freq", 1)

        if device == 'auto':
            device, net = btorch.utils.trainer.auto_gpu(net)

        net.to(device)
        if resume_path is not None:
            start_epoch = resume(resume_path, net, optimizer, lr_scheduler)

        train_loss_data = []
        test_loss_data = []

        for epoch in range(start_epoch, max_epoch):
            train_loss = cls.train_epoch(net, criterion, trainloader, optimizer, epoch, device, config)
            train_loss_data.append(train_loss)
            test_loss = "Not Provided"
            if testloader is not None and epoch%val_freq==0:
                test_loss = cls.test_epoch(net, criterion, testloader, device, config)
                test_loss_data.append(test_loss)
            if save_path is not None:
                to_save = dict(train_loss_data=train_loss_data,
                            test_loss_data=test_loss_data,
                            config=config)
                if test_loss <= min(test_loss_data, default=999):
                    save_model(net, f"{save_path}_best.pt",
                            to_save, optimizer, lr_scheduler)
                save_model(net, f"{save_path}_latest.pt",
                        to_save, optimizer, lr_scheduler)
                if epoch % 20 == 0:
                    save_model(net, f"{save_path}_{epoch}.pt",
                            to_save, optimizer, lr_scheduler)
            print(f"Epoch {epoch}: Training loss: {train_loss}. Testing loss: {test_loss}")
            if lr_scheduler is not None:
                lr_scheduler.step()
        return dict(train_loss_data=train_loss_data,
                    test_loss_data=test_loss_data)

    @classmethod
    def train_epoch(cls, net, criterion, trainloader, optimizer, epoch_idx, device='cuda', config=None):
        """This is the very basic training function for one epoch. Override this function when necessary
            
        Returns:
            (float or dict): train_loss
        """
        net.train()
        train_loss = 0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predicted = net(inputs)
            loss = criterion(predicted, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pbar.set_description(
                f"epoch {epoch_idx+1} iter {batch_idx}: train loss {loss.item():.5f}.")
        return train_loss/(batch_idx+1)

    @classmethod
    def test_epoch(cls, net, criterion, testloader, device='cuda', config=None):
        """This is the very basic evaluating function for one epoch. Override this function when necessary
            
        Returns:
            (float or dict): eval_loss
        """
        net.eval()
        test_loss = 0
        with torch.inference_mode():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                predicted = net(inputs)
                loss = criterion(predicted, targets)
                test_loss += loss.item()
        return test_loss/(batch_idx+1)

    @classmethod
    def predict_(cls, net, loader, device='cuda', config=None):
        """This is the very basic predicting function. Override this function when necessary
            
        Returns:
            (list or dict): predict results
        """
        net.to(device)
        net.eval()
        out = []
        with torch.inference_mode():
            for batch_idx, (inputs, _) in enumerate(loader):
                inputs =  inputs.to(device)
                out.append(net(inputs))
        return out

    @classmethod
    def overfit_small_batch_(cls, net, criterion, dataset, optimizer, config=None):
        """This is a helper function to check if your model is working by checking if it can overfit a small dataset.
        Note: This function will affect the model weights and all other training-related setting/parameters.
        It uses .train_epoch().
        """
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise ValueError("Currently only support Dataset as input")
        dataset = torch.utils.data.Subset(dataset, [0,1,2,3])
        loader = DataLoader(dataset,2)
        loss_history = []
        for epoch in range(100):
            train_loss = cls.train_epoch(net, criterion, loader, optimizer, epoch, config['device'], config)
            loss_history.append(train_loss)
        print(loss_history)
        # del net_test
        try:
            last_loss = loss_history[-1].item()
            if last_loss < 1e-5:
                print("It looks like your model is working. Please double check the loss_history to see whether it is overfitting. Expected to be overfit.")
        except:
            pass
        print("Please check the loss_history to see whether it is overfitting. Expected to be overfit.")

    def cuda(self, device=None):
        self._config['device'] = 'cuda'
        return self._apply(lambda t: t.cuda(device))

    def set_gpu(self):
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available but you are setting the model to GPU mode.")
        self._config['device'] = 'cuda'
        self.to('cuda')

    def cpu(self):
        self.set_cpu()

    def set_cpu(self):
        self._config['device'] = 'cpu'
        self.to('cpu')

    def auto_gpu(self, parallel='auto', on=None):
        device, _ = btorch.utils.trainer.auto_gpu(self, parallel, on)
        self._config['device'] = device
    
    def device(self):
        return next(self.parameters()).device

    def save(self, filepath, include_optimizer=True, include_lr_scheduler=True):
        """Saves the model.state_dict and self._history.

        Args:
            filepath (str): PATH
        """
        to_save_optim = self._optimizer if include_optimizer else None
        to_save_lrs = self._lr_scheduler if include_lr_scheduler else None
        save_model(self, filepath, self._history, optimizer=to_save_optim,
                   lr_scheduler=to_save_lrs)

    def load(self, filepath, skip_mismatch=False):
        """Load the model.state_dict.

        Args:
            filepath (str): PATH
        """
        state = torch.load(filepath)
        self.load_state_dict(state['model'], not skip_mismatch)

    def summary(self, *args, **kwargs):
        """Prints a string summary of network. https://github.com/TylerYep/torchinfo
        """
        return summary(self, *args, **kwargs)
    
    def number_parameters(self, exclude_freeze=False):
        """Returns the number of parameters in the model.
        """
        return btorch.utils.number_params(self, exclude_freeze)

class GridSearch():
    def __init__(self, model, base_config, param_grid, scoring=None, **kwargs):
        self.model = model
        self.base_config = base_config
        self.param_grid = param_grid
        self.scoring = scoring
        if self.scoring is None:
            self.scoring = btorch.utils.accuracy_score
        self.log = {}
        self.best_model = None
        self.best_score = None
        if kwargs['_lossfn'] is None or kwargs['_optimizer'] is None:
            raise Exception('`_lossfn` and `_optimizer` is not set.')
        self._lossfn = kwargs['_lossfn']
        self._optimizer = kwargs['_optimizer']
        if kwargs['_lr_scheduler']:
            self._lr_scheduler = kwargs['_lr_scheduler']
        else:
            self._lr_scheduler = None

    def init_model(self, curr_config, *args, **kwargs):
        model = self.model(**{**self.base_config, **curr_config})
        model._lossfn = self._lossfn
        model._optimizer = self._optimizer
        model._lr_scheduler = self._lr_scheduler


    def score(self, net, x, y):
        y_pred = net.predict(x, return_combined=True)
        return self.scoring(y_pred, y)

    def fit(self, x=None, y=None, **kwargs):
        for curr_config in self.param_grid:
            curr_model = self.init_model(curr_config)
            curr_model.fit(x, y, **kwargs)
            score = self.score(curr_model, x, y)
            self.log[str({**self.base_config, **curr_config})] = score
            if self.best_score is None:
                self.best_model = curr_model
                self.best_score = score
            elif score > self.best_score:
                self.best_model = curr_model
                self.best_score = score

                        



    