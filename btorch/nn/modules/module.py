import math
import warnings

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchinfo import summary
from tqdm import tqdm

import btorch
from btorch.utils.load_save import save_model, resume


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

    If you decided to use the highlevel training loop. Please set the following instance attributes.
    You can set the attributes to anything you want for advanced use.
    The default attributes guideline is only for the default highlevel functions.
    
    Attributes:
      self._lossfn (default to pytorch Loss Func): **Required**. ``criterion`` in @classmethod
      self._optimizer (default to pytorch Optimizer): **Required**.``optimizer`` in @classmethod
      self._lr_scheduler (default to pytorch lr_Scheduler): **Optional**. ``lr_scheduler`` in @classmethod
      self._history (default to list): **Optional**. All loss, evaluation metric should be here.
      self._config (default to dict): **Optional**. ``config`` in @classmethod.
        Contains all setting and hyper-parameters for training loops
        
        For Defaults usage, it accepts:
            - start_epoch (int): start_epoch idx. Defaults to 0.
            - max_epoch (int): max number of epoch. Defaults to 10.
            - device (str): either 'cuda' or 'cpu' or 'auto'. Defaults to 'cpu'.
            - save (str): save model path. Save the best (lowest loss) and latest model. Defaults to None.
            - resume (str): resume model path. Override start_epoch. Defaults to None.
            - save_every_epoch_checkpoint (int): Enable save the best model and every x epoch.
                Default to None, it will not save on every epoch.
            - save_base_on (str): only useful when ``cls.test_epoch()`` returns a dict.
                The key of the dict to determine the best model.
                Defaults to 'loss'.
            - val_freq (int): freq of running validation. Defaults to 1.
            - tensorboard (str or bool): Enable logging to Tensorboard. Defaults to None.
                Input the session(log_dir)_name(``str``) or ``True`` to enable.
                Run ``$tensorboard --logdir=runs`` on terminal to start Tensorboard.
      
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
            "save_base_on": 'loss',
            "val_freq": 1,
            "tensorboard": None,
        }

    def fit(self, x=None, y=None, batch_size=8, epochs=None, shuffle=True, drop_last=False,
            validation_split=0.0, validation_data=None, validation_batch_size=8, validation_freq=None,
            scoring=None, initial_epoch=None, workers=1, **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).
        
        Keras like fit method. All arguments follow `Keras usage
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`__.
        It uses .train_net()

        Args:
            x: Input data. It could be
                - torch.tensor in batch node, starting with (N, *)
                - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, targets)``
                - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored, if provided.
            y: Target data. Like the input data ``x``,
              it should be torch.Tensor.
              If ``x`` is a dataset, generator or dataloader, ``y`` should
              not be specified (since targets will be obtained from ``x``).
            batch_size (int, optional): Defaults to 10.
            epochs (int, optional): max_epochs. Defaults to 1.
            shuffle (bool, optional): Shuffle the data or not. Defaults to True.
            drop_last (bool, optional): All batch has same shape or not. Defautls to False.
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
            validation_batch_size (optional): batch size of validation data.
            validation_freq (optional): runs validation every x epochs.
            scoring (Callable, optional): A scoring function that take in ``y_true`` and ``model_output``
              Usually, this is your evaluation metric, like accuracy.
              If provided, this method return a dict that include both loss and score.
              This scoring function should return the **sum** (set ``reduction=sum``) of the score of a batch.
              This will only apply to validation data by default.
              The function signature must be ``scoring(y_true=, model_output=)``.
            initial_epoch (optional): start epoch. Return from ``btorch.utils.load_save.resume``
            workers (optional): num_workers for dataloader
            
        Kwargs:
            - verbose (optional): verbose level. 0 means print nothings. Defaults to 1.

        Note:
            x should be a DataLoader/Dataset that yields two element (inputs, targets).
            If your x yields more than 2 elements or you input x as list of Tensor or y as list of Tensor.
            You might need to manually change the ``for batch_idx, (inputs, targets) in pbar:``
            in each classmethod (eg. @train_epoch).
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
        if isinstance(x, torch.Tensor) or isinstance(x, (tuple, list)):
            assert y is not None, f"x is {type(x)}, expected y to be torch.Tensor or List[Tensor]"
            x = btorch.utils.tensor_to_Dataset(x, y)
        elif isinstance(x, (torch.utils.data.DataLoader, torch.utils.data.Dataset)) and y is not None:
            warnings.warn(f"x is {type(x)}, y should be not specified and will be ignored.")

        # Pre-process eval_data
        if validation_data is not None:
            if isinstance(validation_data, (tuple, list)):
                assert len(validation_data) == 2, "``validation_data`` should have only 2 element, [eval_x, eval_y]."
                x_eval = btorch.utils.tensor_to_Dataset(validation_data[0], validation_data[1])
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
                                            trainloader=train_loader, testloader=eval_loader, scoring=scoring,
                                            lr_scheduler=self._lr_scheduler, config=self._config, **kwargs))

    def evaluate(self, x=None, y=None, batch_size=8, scoring=None, drop_last=False, workers=1, **kwargs):
        """Returns the loss value & metrics values for the model in test mode.

        Keras like evaluate method. All arguments follows Keras usage.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
       

        Args:
            x: Input data. It could be
                - torch.tensor in batch node, starting with (N, *)
                - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, targets)``
                - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored, if provided.
            y: Target data. Like the input data ``x``,
              It should be torch.tensor.
              If ``x`` is a dataset, generator or dataloader, ``y`` should
              not be specified (since targets will be obtained from ``x``).
            batch_size (int, optional): Defaults to 8.
            scoring (Callable, optional): A scoring function that take in ``y_true`` and ``model_output``
              Usually, this is your evaluation metric, like accuracy.
              If provided, this method return a dict that include both loss and score.
              This scoring function should return the **sum**(set ``reduction=sum``) of the score of a batch.
              The function signature must be ``scoring(y_true=, model_output=)``..
            drop_last (bool, optional): Shuffle the data or not. 
            workers (optional): num_workers for dataloader

        Returns:
            (float or dict)
        
        Note:
            It uses .test_epoch()
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
        return self.test_epoch(net=self, criterion=self._lossfn, testloader=test_loader, scoring=scoring,
                               epoch_idx=0, device=self._config.get("device", "cpu"), config=self._config)

    def predict(self, x, batch_size=8, return_combined=False):
        """Generates output predictions for input samples.

        Keras like predict method. All arguments follows `Keras usage.
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`__.

        Args:
            x: Input data. It could be
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
            
        Note:
            It uses .predict_()
        """
        if isinstance(x, torch.Tensor):
            _y = torch.zeros(x.shape[0])
            dataset = TensorDataset(x, _y)
        elif isinstance(x, torch.utils.data.Dataset):
            dataset = x
        elif isinstance(x, torch.utils.data.DataLoader):
            pass
        else:
            raise ValueError(f"x is {type(x)}, it should be ``Tensor``, ``Dataset`` or ``DataLoader``")
        if isinstance(x, torch.utils.data.DataLoader):
            loader = x
        else:
            loader = DataLoader(dataset, batch_size)
        out = self.predict_(net=self, loader=loader, device=self._config.get('device', 'cpu'), config=self._config)

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
                        f"Please manually handle the output type from ``self.predict_``, got {type(out[0])}."
                        f"Only support List[Dict] and List[Tensor]. You can also turn off ``return_combined``.")
            else:
                warnings.warn(
                    f"The output type from ``self.predict_`` is {type(out)}, "
                    f"``return_combined`` is only useful when it is ``list``")
                return out
        else:
            return out

    def overfit_small_batch(self, x):
        self.overfit_small_batch_(self, self._lossfn, x, self._optimizer, self._config)

    @classmethod
    def train_net(cls, net, criterion, optimizer, trainloader, testloader=None, lr_scheduler=None, scoring=None,
                  config=None, **kwargs):
        """Standard PyTorch training loop. Override this function when necessary.
        
        Args:
          net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
          criterion (any): can be a loss function or a list/dict of loss functions. It will be used in ``@train_epoch``
          optimizer ([type]): can be a optimizer or a list/dict of optimizers. It will be used in ``@train_epoch``
          trainloader (torch.utils.data.Dataloader): can be a dataloader or a list/dict of dataloaders. It will be used in ``@train_epoch``
          testloader (torch.utils.data.Dataloader, optional): can be a dataloader or a list/dict of dataloaders. It will be used in ``@train_epoch``. Defaults to None.
          lr_scheduler (torch.optim.lr_scheduler, optional): Defaults to None.
          scoring (Callable, optional): A scoring function that take in ``y_true`` and ``model_output``
              Usually, this is your evaluation metric, like accuracy.
              If provided, this method return a dict that include both loss and score.
              This scoring function should return the **sum**(set ``reduction=sum``) of the score of a batch.
              This will only apply to .test_epoch() by default.
              The function signature is ``scoring(y_true=, model_output=)``.
          config (dict, optional): Config for training. Defaults to None.
          
        Note:
          It uses .train_epoch() and .test_epoch()
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
        tensorboard_writer = config.get("tensorboard", None)
        if config.get("tensorboard", None) is True or isinstance(config.get("tensorboard", None), str):
            from torch.utils.tensorboard import SummaryWriter
            name = f"runs/{config.get('tensorboard', None)}" if isinstance(config.get("tensorboard", None),
                                                                           str) else None
            tensorboard_writer = SummaryWriter(log_dir=name)
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
            cls.before_each_train_epoch(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader,
                                        testloader=testloader, epoch_idx=epoch, lr_scheduler=lr_scheduler,
                                        config=config, **kwargs)
            train_loss = cls.train_epoch(net=net, criterion=criterion, trainloader=trainloader,
                                         optimizer=optimizer, epoch_idx=epoch, device=device, config=config, **kwargs)
            train_loss_data.append(train_loss)
            cls.add_tensorboard_scalar(tensorboard_writer, 'train_loss', train_loss, epoch)
            test_loss = "Not Provided"
            if testloader is not None and epoch % val_freq == 0:
                test_loss = cls.test_epoch(net=net, criterion=criterion, testloader=testloader, scoring=scoring,
                                           epoch_idx=epoch, device=device, config=config, **kwargs)
                test_loss_data.append(test_loss)
                cls.add_tensorboard_scalar(tensorboard_writer, 'test_loss', test_loss, epoch)
            if save_path is not None:
                to_save = dict(train_loss_data=train_loss_data,
                               test_loss_data=test_loss_data,
                               config=config,
                               epoch=epoch)
                save_model(net, f"{save_path}_latest.pt",
                           to_save, optimizer, lr_scheduler)
                if test_loss != "Not Provided":
                    if isinstance(test_loss, dict) and test_loss[config.get('save_base_on', 'loss')] <= \
                            min(test_loss_data, key=lambda x: x[config.get('save_base_on', 'loss')], default=999)[
                                config.get('save_base_on', 'loss')]:
                        save_model(net, f"{save_path}_best.pt",
                                   to_save, optimizer, lr_scheduler)
                    elif not isinstance(test_loss, dict) and test_loss <= min(test_loss_data, default=999):
                        save_model(net, f"{save_path}_best.pt",
                                   to_save, optimizer, lr_scheduler)
                if save_every_epoch_checkpoint is not None:
                    if epoch % save_every_epoch_checkpoint == 0:
                        save_model(net, f"{save_path}_{epoch}.pt",
                                   to_save, optimizer, lr_scheduler)
            if kwargs.get("verbose", 1):
                print(f"Epoch {epoch}: Training loss: {train_loss}. Testing loss: {test_loss}")
            if lr_scheduler is not None:
                lr_scheduler.step()
            cls.after_each_train_epoch(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader,
                                       testloader=testloader, epoch_idx=epoch, lr_scheduler=lr_scheduler,
                                       config=config, **kwargs)
        if config.get("tensorboard", None):
            tensorboard_writer.flush()
        return dict(train_loss_data=train_loss_data,
                    test_loss_data=test_loss_data)

    @classmethod
    def train_epoch(cls, net, criterion, trainloader, optimizer, epoch_idx, device='cuda', config=None, **kwargs):
        """This is the very basic training function for one epoch. Override this function when necessary
            
        Returns:
            (float or dict): train_loss
        """
        net.train()
        train_loss = 0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), disable=(kwargs.get("verbose", 1) == 0))
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
    def test_epoch(cls, net, criterion, testloader, scoring=None, epoch_idx=0, device='cuda', config=None, **kwargs):
        """This is the very basic evaluating function for one epoch. Override this function when necessary

        Args:
            scoring (Callable, optional): A scoring function that take in ``y_true`` and ``model_output``.
              Usually, this is your evaluation metric, like accuracy.
              If provided, this method return a dict that include both loss and score.
              This scoring function should return the **sum** (set ``reduction=sum``) of the score of a batch.
              The function signature must be ``scoring(y_true=, model_output=)``.
              
        Returns:
            (float or dict): eval_loss
        """
        net.eval()
        test_loss = 0
        test_score = 0
        total = 0
        with torch.inference_mode():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                predicted = net(inputs)
                loss = criterion(predicted, targets)
                test_loss += loss.item()
                if scoring is not None:
                    score = scoring(model_output=predicted, y_true=targets)
                    test_score += score
                total += len(inputs)
        if scoring is None:
            return test_loss / (batch_idx + 1)
        return {'loss': test_loss / (batch_idx + 1), 'score': test_score / total}

    @classmethod
    def before_each_train_epoch(cls, net, criterion, optimizer, trainloader, testloader=None, epoch_idx=0,
                                lr_scheduler=None, config=None, **kwargs):
        """You can override this function to do something before each epoch.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

    @classmethod
    def after_each_train_epoch(cls, net, criterion, optimizer, trainloader, testloader=None, epoch_idx=0,
                               lr_scheduler=None, config=None, **kwargs):
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
        
        Note:
            It uses .train_epoch().
            
            This function will affect the model weights and all other training-related setting/parameters.
         
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
            _lossfn (functools.partial Class Constructor): loss function for btorch model.
            _optimizer (functools.partial Class Constructor): optimizer for btorch model.
            _lr_scheduler (functools.partial Class Constructor): lr_scheduler for btorch model.
            
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
