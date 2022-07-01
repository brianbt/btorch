import warnings
from tqdm import tqdm
from torchinfo import summary
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import btorch
from btorch.utils.load_save import save_model, resume
import math


class Module(nn.Module):
    """Base class for all neural network modules.

    Your models should also subclass this class.
    btorch.nn.Module is inhernet from pytorch.nn, hence, all syntax is same as pytorch.
    
    You can replace your ``from torch import nn`` as ``from btorch import nn``.
    
    This class provides some highlevel training method like Keras
      - .fit()
      - .evaluate()
      - .predict()
      - .overfit_small_batch()
    These highlevel method will use the core class methods which should be overrided for advanced use.
    
    The core class methods are
      - .train_net()
      - .train_epoch()
      - .test_epoch()
      - .predict_()
      - .overfit_small_batch_()
    All of above classmethods can be overrided at your need.
    
    Note:
      When overriding instance method, call classmethod via ``self.``
      When overriding class method, remember to put ``@classmethod``.
      If you want to use core class method directly in real use case, use as follow:
        >>> class Model(nn.Module):
        >>>     ...
        >>> mod = Model()
        >>> mod.train_net(...)   # correct
        >>> Model.train_net(...) # wrong
      When overriding class method and if you want to use instance method (eg. fit),
      you should keep the exact SAME signature in the class method. 
      
      Inside class method
        - call instance variable via ``net.``
        - call instance method via ``net.``
        - call class method via ``cls.``

    Hierarchy View:
      | .fit
      | └── @train_net -> {train_loss, test_loss}
      |     ├── @before_each_train_epoch [optional]
      |     ├── @train_epoch -> train_loss
      |     ├── @after_each_train_epoch [optional]
      |     └── @test_epoch -> test_loss [optional]

      | .evaluate -> test_loss
      | └── @test_epoch -> test_loss

      | .predict -> prediction
      | └── @predict_ -> prediction

      | .overfit_small_batch
      | └── @overfit_small_batch_
      |     └── @train_epoch -> train_loss

    If you decided to use the highlevel training loop. Please set the following instance attributes: 
      - self._lossfn (default:pytorch Loss Func)[required][``criterion`` in @classmethod]
      - self._optimizer (default:pytorch Optimizer)[required][``optimizer`` in @classmethod]
      - self._lr_scheduler (default:pytorch lr_Scheduler)[optional][``lr_scheduler`` in @classmethod]
      - self._config (default:dict)[optional][``config`` in @classmethod]
        Contains all setting and hyper-parameters for training loops
        For Defaults usage, it accepts:
          - start_epoch (int): start_epoch idx
          - max_epoch (int): max number of epoch
          - device (str): either ``cuda`` or ``cpu`` or ``auto``
          - save (str): save model path
          - resume (str): resume model path. Override start_epoch
          - save_every_epoch_checkpoint (int): Enable save the best model and every x epoch
          - val_freq (int): freq of running validation
          - tensorboard (SummaryWriter): Enable logging to Tensorboard.
              Input the session(log_dir) name or ``True`` to enable.
              Input a ``SummaryWriter`` object to use it.
              Run ``$tensorboard --logdir=runs`` on terminal to start Tensorboard.
      - self._history (default:list)[optional]. All loss, evaluation metric should be here.
    You can set them to be a pytorch instance (or a dictionary, for advanced uses).
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
        self.__config = dict()
        self._lossfn = None
        self._optimizer = None
        self._lr_scheduler = None
        self.init_config()
        self._history = []

    def init_config(self):
        """Initialize the config to Default.
        """
        self._config = {
            "start_epoch": 0,
            "max_epoch": 10,
            "device": "cpu",
            "save": None,
            "resume": None,
            "save_every_epoch_checkpoint": None,
            "val_freq": 1,
            "tensorboard": None,
        }

    def fit(self, x=None, y=None, batch_size=8, epochs=None, shuffle=True, drop_last=False,
            validation_split=0.0, validation_data=None, validation_batch_size=8, validation_freq=None,
            initial_epoch=None, workers=1):
        """Trains the model for a fixed number of epochs (iterations on a dataset).
        
        Keras like fit method. All arguments follow `Keras usage
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`__.
        It uses .train_net()

        Args:
            x: Input data. It could be:
              - torch.Tensor in batch node, starting with (N, *)
              - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, targets)``
              - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored, if provided.
            y: Target data. Like the input data ``x``,
              it should be torch.Tensor.
              If ``x`` is a dataset, generator or dataloader, ``y`` should
              not be specified (since targets will be obtained from ``x``).
            batch_size (int, optional): Defaults to 10.
            epochs (int, optional): Defaults to 1.
            shuffle (bool, optional): Defaults to True.
            drop_last (bool, optional): Shuffle the data or not. 
            validation_split (optional): Float between 0 and 1.
              Fraction of the training data to be used as validation data.
              The model will set apart this fraction of the training data,
              will not train on it. This argument is
              not supported when ``x`` is a Dataset or Dataloader.
              Defaults to 0.
            validation_data (optional): Data on which to evaluate the loss 
              and any model metrics at the end of each epoch. 
              The model will not be trained on this data. Defaults to None.
              ``validation_data`` will override ``validation_split``. 
              ``validation_data`` could be:
                - tuple of torch.tensor, tuple(X, y)
                - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, targets)``
                - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored.
            validation_batch_size (optional): batch size of validation data
            validation_freq (optional): runs validation every x epochs.
            initial_epoch (optional): start epoch. Return from ``btorch.utils.load_save.resume``
            workers (optional): num_workers for dataloader
        """
        if x is None:
            raise ValueError("x is not provided")
        if self._lossfn is None or self._optimizer is None:
            raise ValueError("``self._lossfn`` and ``self._optimizer`` is not set.")
        # Override config with parameter.
        self._config['max_epoch'] = epochs if epochs is not None else self._config['max_epoch']
        self._config['start_epoch'] = initial_epoch if initial_epoch is not None else self._config['start_epoch']
        self._config['val_freq'] = validation_freq if epochs is not None else self._config['val_freq']

        pin_memory = True if self._config.get('device', 'cpu') == 'cuda' else False
        x_eval = None
        eval_loader = None

        # Pre-process train_data
        if isinstance(x, torch.Tensor):  # Handle if x is tensor
            assert y is not None, f"x is {type(x)}, expected y to be torch.Tensor or List[Tensor]"
            if isinstance(y, (tuple, list)):  # Handle if y is list
                x = TensorDataset(x, *y)
            else:  # Handle if y is tensor
                x = TensorDataset(x, y)
        elif isinstance(x, (tuple, list)):  # Handle if x is list
            assert y is not None, f"x is {type(x)}, expected y to be torch.Tensor or List[Tensor]"
            if isinstance(y, (tuple, list)):  # Handle if y is list
                x = TensorDataset(*x, *y)
            else:  # Handle if y is tensor
                x = TensorDataset(*x, y)
        elif isinstance(x, (torch.utils.data.DataLoader, torch.utils.data.Dataset)):
            warnings.warn(f"x is {type(x)}, y should be not specified and will be ignored.")

        # Pre-process eval_data
        if validation_data is not None:
            if isinstance(validation_data, (tuple, list)):
                assert len(validation_data) == 2, "``validation_data`` should have only 2 element, [eval_x, eval_y]."
                if isinstance(validation_data[0], torch.Tensor):  # Handle if eval_x is tensor
                    if isinstance(y, (tuple, list)):  # Handle if eval_y is list
                        x_eval = TensorDataset(validation_data[0], *validation_data[1])
                    else:  # Handle if eval_y is tensor
                        x_eval = TensorDataset(validation_data[0], validation_data[1])
                elif isinstance(validation_data[0], (tuple, list)):  # Handle if eval_x is list
                    if isinstance(validation_data[1], (tuple, list)):  # Handle if eval_y is list
                        x_eval = TensorDataset(*validation_data[0], *validation_data[1])
                    else:  # Handle if eval_y is tensor
                        x_eval = TensorDataset(*validation_data[0], validation_data[1])
            elif isinstance(validation_data, torch.utils.data.Dataset):
                x_eval = validation_data
            elif isinstance(validation_data, torch.utils.data.DataLoader):
                eval_loader = validation_data
                x_eval = None
            else:
                raise ValueError(f"validation_data doesn't support {type(validation_data)}")
        elif validation_split != 0:
            if isinstance(x, torch.utils.data.DataLoader):
                raise ValueError(f"x is DataLoader, it does not support validation_split.")
            eval_len = math.ceil(validation_split * len(x))
            train_len = len(x) - eval_len
            x, x_eval = torch.utils.data.random_split(x, [train_len, eval_len])

        # Make dataset to dataloader
        if isinstance(x, torch.utils.data.DataLoader):
            train_loader = x
        else:
            train_loader = DataLoader(x, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                      pin_memory=pin_memory, drop_last=drop_last)
        if x_eval is not None:
            eval_loader = DataLoader(x_eval, batch_size=validation_batch_size, num_workers=workers,
                                     pin_memory=pin_memory, drop_last=drop_last)
        # Call @train_net
        self._history.append(self.train_net(net=self, criterion=self._lossfn, optimizer=self._optimizer,
                                            trainloader=train_loader, testloader=eval_loader,
                                            lr_scheduler=self._lr_scheduler, config=self._config))

    def evaluate(self, x=None, y=None, batch_size=None, drop_last=False, workers=1, **kwargs):
        """Returns the loss value & metrics values for the model in test mode.

        Keras like evaluate method. All arguments follows Keras usage.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
        It uses .test_epoch()

        Args:
            x: Input data. It could be:
              - torch.tensor in batch node, starting with (N, *)
              - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, targets)``
              - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored, if provided.
            y: Target data. Like the input data ``x``,
              it should be torch.tensor.
              If ``x`` is a dataset, generator or dataloader, ``y`` should
              not be specified (since targets will be obtained from ``x``).
            batch_size (int, optional): Defaults to 10.
            drop_last (bool, optional): Shuffle the data or not. 
            workers (optional): num_workers for dataloader
            """
        # Pre-process train_loader
        pin_memory = True if self._config.get('device', 'cpu') == 'cuda' else False
        if isinstance(x, torch.utils.data.DataLoader):
            test_loader = x
        else:
            if isinstance(x, torch.Tensor):
                if y is None:
                    raise ValueError(f"x is {type(x)}, expected y to be torch.Tensor")
                x = TensorDataset(x, y)
            elif isinstance(x, (tuple, list)):
                if isinstance(y, (tuple, list)):
                    x = TensorDataset(*x, *y)
                else:
                    x = TensorDataset(*x, y)
            test_loader = DataLoader(x, batch_size=batch_size, num_workers=workers,
                                     pin_memory=pin_memory, drop_last=drop_last)
        return self.test_epoch(self, self._lossfn, test_loader, 0, self._config.get("device", "cpu"))

    def predict(self, x, batch_size=8, return_combined=False):
        """Generates output predictions for input samples.

        Keras like predict method. All arguments follows Keras usage.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        It uses .predict_()

        Args:
            x: Input data. It could be:
              - torch.tensor in batch node, starting with (N, *)
              - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, _)``
              - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored, if provided.
            batch_size (int, optional). Defaults to 8.
            return_combined (bool, optional). 
              - if return from ``self.predict_`` is a list. Combine them into a single object.
              - if return is list of tensor: Apply torch.cat() on the output from .predict_() .
              - if return is list of dict: combined them into one big dict.
              - Defaults to False.

        Returns:
            List[Tensor] or Tensor if return_combined 
        """
        if isinstance(x, torch.Tensor):
            _y = torch.zeros(x.shape[0])
            dataset = TensorDataset(x, _y)
        elif isinstance(x, torch.utils.data.Dataset):
            dataset = x
        elif isinstance(x, torch.utils.data.DataLoader):
            pass
        else:
            raise ValueError("x should be ``Tensor``, ``Dataset`` or ``DataLoader``")
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
                    raise NotImplementedError(
                        f"Please manually handle the output type from ``self.predict_``, got {type(out[0])}. You can also turn off ``return_combined``.")
            else:
                warnings.warn(
                    f"The output type from ``self.predict_`` is {type(out)}, return_combined only useful when it is ``list``")
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
          net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
          criterion (any): can be a loss function or a list/dict of loss functions. It will be used in ``@train_epoch``
          optimizer ([type]): can be a optimizer or a list/dict of optimizers. It will be used in ``@train_epoch``
          trainloader (torch.utils.data.Dataloader): can be a dataloader or a list/dict of dataloaders. It will be used in ``@train_epoch``
          testloader (torch.utils.data.Dataloader, optional): can be a dataloader or a list/dict of dataloaders. It will be used in ``@train_epoch``. Defaults to None.
          lr_scheduler (torch.optim.lr_scheduler, optional): Defaults to None.
          config (dict, optional): Config for training. Defaults to None.
        """
        # Handle config parameters
        if config is None:
            config = dict()
        start_epoch = config.get("start_epoch", 0)
        max_epoch = config.get("max_epoch", 10)
        device = config.get("device", "cpu")
        save_path = config.get("save", None)
        resume_path = config.get("resume", None)
        save_every_epoch_checkpoint = config.get("save_every_epoch_checkpoint", None)
        val_freq = config.get("val_freq", 1)
        if config.get("tensorboard", None) is True or isinstance(config.get("tensorboard", None), str):
            from torch.utils.tensorboard import SummaryWriter
            name = f"runs/{config.get('tensorboard', None)}" if isinstance(config.get("tensorboard", None),
                                                                           str) else None
            config['tensorboard'] = SummaryWriter(log_dir=name)
        if save_every_epoch_checkpoint is not None and save_path is None:
            warnings.warn(
                "``save_every_epoch_checkpoint`` is set, but ``save_path`` is not set. It will not save any checkpoint.")

        # Set GPU and resume
        if device == 'auto':
            device, net = btorch.utils.trainer.auto_gpu(net)
        net.to(device)
        if resume_path is not None:
            start_epoch = resume(resume_path, net, optimizer, lr_scheduler)

        train_loss_data = []
        test_loss_data = []
        # Training Loop
        for epoch in range(start_epoch, max_epoch):
            cls.before_each_train_epoch(net, criterion, optimizer, trainloader, testloader, epoch, lr_scheduler, config)
            train_loss = cls.train_epoch(net, criterion, trainloader, optimizer, epoch, device, config)
            train_loss_data.append(train_loss)
            cls.add_tensorboard_scalar(config.get("tensorboard", None), 'train_loss', train_loss, epoch)
            test_loss = "Not Provided"
            if testloader is not None and epoch % val_freq == 0:
                test_loss = cls.test_epoch(net, criterion, testloader, epoch, device, config)
                test_loss_data.append(test_loss)
                cls.add_tensorboard_scalar(config.get("tensorboard", None), 'test_loss', test_loss, epoch)
            if save_path is not None:
                to_save = dict(train_loss_data=train_loss_data,
                               test_loss_data=test_loss_data,
                               config=config)
                save_model(net, f"{save_path}_latest.pt",
                           to_save, optimizer, lr_scheduler)
                if save_every_epoch_checkpoint is not None:
                    if test_loss <= min(test_loss_data, default=999):
                        save_model(net, f"{save_path}_best.pt",
                                   to_save, optimizer, lr_scheduler)
                    if epoch % save_every_epoch_checkpoint == 0:
                        save_model(net, f"{save_path}_{epoch}.pt",
                                   to_save, optimizer, lr_scheduler)
            print(f"Epoch {epoch}: Training loss: {train_loss}. Testing loss: {test_loss}")
            if lr_scheduler is not None:
                lr_scheduler.step()
            cls.after_each_train_epoch(net, criterion, optimizer, trainloader, testloader, epoch, lr_scheduler, config)
        if config.get("tensorboard", None):
            config.get("tensorboard", None).flush()
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
        batch_idx = 1
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predicted = net(inputs)
            loss = criterion(predicted, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pbar.set_description(
                f"epoch {epoch_idx + 1} iter {batch_idx}: train loss {loss.item():.5f}.")
        return train_loss / (batch_idx + 1)

    @classmethod
    def test_epoch(cls, net, criterion, testloader, epoch_idx=0, device='cuda', config=None):
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
        return test_loss / (batch_idx + 1)

    @classmethod
    def before_each_train_epoch(cls, net, criterion, optimizer, trainloader, testloader=None, epoch=0,
                                lr_scheduler=None, config=None):
        """You can override this function to do something before each epoch.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

    @classmethod
    def after_each_train_epoch(cls, net, criterion, optimizer, trainloader, testloader=None, epoch=0, lr_scheduler=None,
                               config=None):
        """You can override this function to do something after each epoch.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

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
                inputs = inputs.to(device)
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
        dataset = torch.utils.data.Subset(dataset, [0, 1, 2, 3])
        loader = DataLoader(dataset, 2)
        loss_history = []
        for epoch in range(100):
            train_loss = cls.train_epoch(net, criterion, loader, optimizer, epoch, config['device'], config)
            loss_history.append(train_loss)
        print(loss_history)
        # del net_test
        try:
            last_loss = loss_history[-1].item()
            if last_loss < 1e-5:
                print(
                    "It looks like your model is working. Please double check the loss_history to see whether it is overfitting. Expected to be overfit.")
        except Exception:
            pass
        print("Please check the loss_history to see whether it is overfitting. Expected to be overfit.")

    @classmethod
    def add_tensorboard_scalar(cls, writer, tag, data, step, *args, **kwargs):
        """One line code for adding data to tensorboard.
        Args:
            writer (SummaryWriter): the writer object.
              Put ``config['tensorboard']`` to this argument.
              If input is None, this function will not do anything.
            tag (str): Name of this data
            data (Tensor or dict): the data to add.
            step (int): the step of the data.
        """
        if writer is not None:
            if isinstance(data, dict):
                writer.add_scalars(tag, data, step)
            else:
                writer.add_scalar(tag, data, step)

    def cuda(self, device=None):
        self._config['device'] = 'cuda'
        return super().cuda(device)

    def set_gpu(self):
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available but you are setting the model to GPU mode. This will change the "
                          "._config['device'] to cuda even though you might recieve an Exception")
        self._config['device'] = 'cuda'
        self.to('cuda')

    def cpu(self):
        self._config['device'] = 'cpu'
        return super().cpu()

    def set_cpu(self):
        self.cpu()

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

    @property
    def _config(self):
        return self.__config

    @_config.setter
    def _config(self, d):
        if isinstance(d, dict):
            self.__config.update(d)
        else:
            raise ValueError("``_config`` must be a dict")


class GridSearchCV:
    def __init__(self, model, param_grid, scoring=None, cv=2, **kwargs):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        if self.scoring is None:
            self.scoring = btorch.utils.accuracy_score
        self.cv = cv
        self.cv_results_ = {'params': []}
        for i in range(cv):
            self.cv_results_[f'split{i}_train_score'] = []
            self.cv_results_[f'split{i}_test_score'] = []
        self.best_model_ = None
        self.best_score_ = None
        self.best_params_ = None

        self._config = dict()
        self._lossfn = None
        self._optimizer = None
        self._lr_scheduler = None

    def all_combination_in_dict_of_list(self, dict_of_list):
        """ Get all combination from a dict of list.
        https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
        """
        import itertools
        keys, values = zip(*dict_of_list.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts

    def init_model(self, curr_config, *args, **kwargs):
        model = self.model(**curr_config)
        model._lossfn = self._lossfn
        model._optimizer = self._optimizer(model.parameters())
        model._lr_scheduler = self._lr_scheduler(model._optimizer)
        if self._config is not None:
            model._config = self._config
        return model

    def score(self, net, x, y):
        y_pred = net.predict(x, return_combined=True)
        return self.scoring(y_pred, y)

    def fit(self, x=None, y=None, val_x=None, val_y=None, **kwargs):
        # Split into ``cv`` folds
        split_num = [len(x)//self.cv for i in range(self.cv-1)]
        split_num.append(len(x) - self.cv)
        splited = torch.utils.data.random_split(x, split_num)

        for curr_params in self.all_combination_in_dict_of_list(self.param_grid):
            for curr_split in range(self.cv):
                curr_model = self.init_model(curr_params)
                curr_model.fit(x, y, **kwargs)
                train_score = self.score(curr_model, x, y)
                test_score = torch.nan
                self.cv_results_['params'].append(str(curr_params))
                self.cv_results_['train_score'].append(train_score)
                if val_x is not None and val_y is not None:
                    test_score = self.score(curr_model, val_x, val_y)
                    self.cv_results_['test_score'].append(test_score)
                if self.best_score_ is None:
                    self.best_model_ = curr_model
                    self.best_score_ = test_score
                    self.best_params_ = curr_params
                elif test_score > self.best_score_:
                    self.best_model_ = curr_model
                    self.best_score_ = test_score
                    self.best_params_ = curr_params
