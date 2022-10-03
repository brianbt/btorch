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
from btorch.utils.trainer import get_lr


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
    All above classmethods can be overrided at your need.

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
      |     ├── @on_train_begin [optional]
      |     ├── @on_train_epoch_begin [optional]
      |     ├──> @train_epoch -> train_loss
      |     ├── @on_train_epoch_end [optional]
      |     ├── @on_test_epoch_begin [optional]
      |     ├──> @test_epoch -> test_loss [optional]
      |     ├── @on_test_epoch_end [optional]
      |     └── @on_train_end [optional]

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
      self._optimizer (default to pytorch Optimizer): **Required**. ``optimizer`` in @classmethod
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
      - .device
      - .number_parameters()

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
        self.train_dataloader = None
        self.eval_dataloader = None
        self.test_dataloader = None
        self.predict_dataloader = None

    def init_config(self):
        """Initialize the config to Default."""
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
    def compile(self, optimizer, lossfn, lr_scheduler=None):
        """Keras like compile function."""
        self._optimizer = optimizer
        self._lossfn = lossfn
        self._lr_scheduler = lr_scheduler

    def fit(self, x=None, y=None, batch_size=8, epochs=None, shuffle=True, drop_last=False,
            validation_split=0.0, validation_data=None, validation_batch_size=8, validation_freq=None,
            scoring=None, initial_epoch=None, workers=1, **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Keras like fit method. All arguments follow `Keras usage
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`__.

        Args:
            x: Input data. 
              It could be:
                - torch.tensor in batch node, starting with (N, *)
                - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, targets)``
                - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored, if provided.
                - if input unsupported type, it will be treated as ``torch.utils.data.Dataset``.
            y: Target data. Like the input data ``x``,
              it should be torch.Tensor.
              If ``x`` is a dataset, generator or dataloader, ``y`` should
              not be specified (since targets will be obtained from ``x``).
            batch_size (int, optional): Defaults to 8.
            epochs (int, optional): max_epochs. Defaults to 10.
            shuffle (bool, optional): Shuffle the data or not. Defaults to True.
            drop_last (bool, optional): All batch has same shape or not. Defautls to False.
            validation_split (optional): Float between 0 and 1.
              Fraction of the training data to be used as validation data.
              The model will set apart this fraction of the training data,
              will not train on it. This argument is
              not supported when ``x`` is a Dataloader.
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
            validation_freq (optional): runs validation every x epochs. Defaults to 1.
            scoring (Callable, optional): A scoring function that take in ``y_true`` and ``model_output``
              Usually, this is your evaluation metric, like accuracy.
              If provided, this method return a dict that include both loss and score.
              This scoring function should return the **sum** (set ``reduction=sum``) of the score of a batch.
              This will only apply to validation data by default.
              The function signature must be ``scoring(y_true=, model_output=)``.
            initial_epoch (optional): start epoch. 
              Return from ``btorch.utils.load_save.resume`` if possible. Defaults to 1.
            workers (optional): num_workers for dataloader. Defaults to 1.

        Kwargs:
            - verbose (optional): verbose level. 0 means print nothings. Defaults to 1.

        Note:
            x should be a DataLoader/Dataset that yields two element (inputs, targets).
            If your x yields more than 2 elements or you input x as list of Tensor or y as list of Tensor.
            You might need to manually change the ``for batch_idx, (inputs, targets) in pbar:``
            in each classmethod (eg. @train_epoch).
            
        Note:
            It uses .train_net()
        """
        if x is None:
            raise ValueError("x is not provided")
        if self._lossfn is None or self._optimizer is None:
            raise ValueError(
                "``self._lossfn`` and ``self._optimizer`` is not set.")
        # Override config with parameter.
        self._config['max_epoch'] = epochs if epochs is not None else self._config['max_epoch']
        self._config['start_epoch'] = initial_epoch if initial_epoch is not None else self._config['start_epoch']
        self._config['val_freq'] = validation_freq if validation_freq is not None else self._config['val_freq']
        if isinstance(scoring, str):
            scoring = btorch.metrics.metrics._get_metric_str(scoring)
        pin_memory = True if self._config.get('device', 'cpu') == 'cuda' else False
        x_eval = None
        eval_loader = None

        # Pre-process train_data
        if isinstance(x, torch.Tensor) or isinstance(x, (tuple, list)):
            assert y is not None, f"x is {type(x)}, expected y to be torch.Tensor or List[Tensor]"
            x = btorch.utils.tensor_to_Dataset(x, y)
        elif isinstance(x, (torch.utils.data.DataLoader, torch.utils.data.Dataset)) and y is not None:
            warnings.warn(
                f"x is {type(x)}, y should be not specified and will be ignored.")
        else:
            warnings.warn(
                f"x might not support {type(x)}. It will treat x as ``Dataset``.")

        # Pre-process eval_data
        if validation_data is not None:
            if isinstance(validation_data, (tuple, list)):
                assert len(
                    validation_data) == 2, "``validation_data`` should have only 2 element, [eval_x, eval_y]."
                x_eval = btorch.utils.tensor_to_Dataset(
                    validation_data[0], validation_data[1])
            elif isinstance(validation_data, torch.utils.data.Dataset):
                x_eval = validation_data
            elif isinstance(validation_data, torch.utils.data.DataLoader):
                eval_loader = validation_data
                x_eval = None
            else:
                warnings.warn(
                    f"validation_data might not support {type(validation_data)}. It will treat validation_data as ``Dataset``.")
                x_eval = validation_data
        elif validation_split != 0:
            if isinstance(x, torch.utils.data.DataLoader):
                raise ValueError(
                    f"x is DataLoader, it does not support validation_split.")
            eval_len = math.ceil(validation_split * len(x))
            train_len = len(x) - eval_len
            x, x_eval = torch.utils.data.random_split(x, [train_len, eval_len])

        # Make dataset to dataloader
        if isinstance(x, torch.utils.data.DataLoader):
            train_loader = x
        else:
            train_loader = DataLoader(x, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                      pin_memory=pin_memory, drop_last=drop_last)
        self.train_dataloader = train_loader
        if x_eval is not None:
            eval_loader = DataLoader(x_eval, batch_size=validation_batch_size, num_workers=workers,
                                     pin_memory=pin_memory, drop_last=drop_last)
            self.eval_dataloader = eval_loader
        # Call @train_net
        self._history.append(self.train_net(net=self, criterion=self._lossfn, optimizer=self._optimizer,
                                            trainloader=train_loader, testloader=eval_loader, scoring=scoring,
                                            lr_scheduler=self._lr_scheduler, config=self._config, **kwargs))

    def evaluate(self, x=None, y=None, batch_size=8, scoring=None, drop_last=False, workers=1, **kwargs):
        """Returns the loss value & metrics values for the model in test mode.

        Keras like evaluate method. All arguments follows Keras usage.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate

        Args:
            x: Input data. 
              It could be:
                - torch.tensor in batch node, starting with (N, *)
                - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, targets)``
                - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored, if provided.
                - if input unsupported type, it will be treated as ``torch.utils.data.Dataset``.
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
        pin_memory = True if self._config.get('device', 'cpu') == 'cuda' else False
        # Pre-process train_data
        if isinstance(x, torch.Tensor) or isinstance(x, (tuple, list)):
            assert y is not None, f"x is {type(x)}, expected y to be torch.Tensor or List[Tensor]"
            x = btorch.utils.tensor_to_Dataset(x, y)
        elif isinstance(x, (torch.utils.data.DataLoader, torch.utils.data.Dataset)) and y is not None:
            warnings.warn(
                f"x is {type(x)}, y should be not specified and will be ignored.")
        else:
            warnings.warn(
                f"x might not support {type(x)}. It will treat x as ``Dataset``.")

        # Make dataset to dataloader
        if isinstance(x, torch.utils.data.DataLoader):
            test_loader = x
        else:
            test_loader = DataLoader(x, batch_size=batch_size, num_workers=workers,
                                     pin_memory=pin_memory, drop_last=drop_last)
        self.test_dataloader = test_loader
        return self.test_epoch(net=self, criterion=self._lossfn, testloader=test_loader, scoring=scoring,
                               epoch_idx=0, device=self._config.get("device", "cpu"), config=self._config)

    def predict(self, x, batch_size=8, return_combined=False):
        """Generates output predictions for input samples.

        Keras like predict method. All arguments follows `Keras usage.
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`__.

        Args:
            x: Input data. 
              It could be:
                - torch.tensor in batch node, starting with (N, *)
                - a ``torch.utils.data.Dataset`` dataset. Should return a tuple of ``(inputs, _)``
                - a ``torch.utils.data.Dataloader``. All other dataset related argument will be ignored, if provided.
                - if input unsupported type, it will be treated as ``torch.utils.data.Dataset``.
            batch_size (int, optional). Defaults to 8.
            return_combined (bool, optional). 
              - if return from ``self.predict_`` is a list. Combine them into a single object.
              - if return is list of tensor: Apply ``torch.cat()`` on the output from ``.predict_()``.
              - if return is list of dict: combined them into one big dict.
              
              Note:
                Suggest to set ``return_combined=False`` and apply 
                ``.dict_operator()`` on the return output for flexibility.
              - Defaults to False.

        Returns:
            List[Tensor] or List[Dict] or Tensor if return_combined 

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
            warnings.warn(
                f"x might not support {type(x)}. It will treat x as ``Dataset``.")
            dataset = x
        if isinstance(x, torch.utils.data.DataLoader):
            loader = x
        else:
            loader = DataLoader(dataset, batch_size=batch_size)
        self.predict_dataloader = loader
        out = self.predict_(net=self, loader=loader, device=self._config.get(
            'device', 'cpu'), config=self._config)

        if return_combined:
            if isinstance(out, list):
                if isinstance(out[0], dict):
                    return btorch.utils.dict_operator.dict_combine(out)
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
        self.overfit_small_batch_(
            self, self._lossfn, x, self._optimizer, self._config, verbose=0)

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
        save_every_epoch_checkpoint = config.get(
            "save_every_epoch_checkpoint", None)
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
        cls.on_train_begin(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader,
                           testloader=testloader, lr_scheduler=lr_scheduler, config=config, **kwargs)
        # Training Loop
        for epoch in range(start_epoch, max_epoch):
            cls.on_train_epoch_begin(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader,
                                     testloader=testloader, epoch_idx=epoch, lr_scheduler=lr_scheduler,
                                     config=config, **kwargs)
            train_loss = cls.train_epoch(net=net, criterion=criterion, trainloader=trainloader,
                                         optimizer=optimizer, epoch_idx=epoch, device=device, config=config, **kwargs)
            cls.on_train_epoch_end(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader,
                                   testloader=testloader, epoch_idx=epoch, lr_scheduler=lr_scheduler,
                                   config=config, **kwargs)
            train_loss_data.append(train_loss)
            cls.add_tensorboard_scalar(
                tensorboard_writer, 'train_loss', train_loss, epoch)
            test_loss = "Not Provided"
            if testloader is not None and epoch % val_freq == 0:
                cls.on_test_epoch_begin(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader,
                                        testloader=testloader, epoch_idx=epoch, lr_scheduler=lr_scheduler,
                                        config=config, **kwargs)
                test_loss = cls.test_epoch(net=net, criterion=criterion, testloader=testloader, scoring=scoring,
                                           epoch_idx=epoch, device=device, config=config, **kwargs)
                cls.on_test_epoch_end(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader,
                                      testloader=testloader, epoch_idx=epoch, lr_scheduler=lr_scheduler,
                                      config=config, **kwargs)
                test_loss_data.append(test_loss)
                cls.add_tensorboard_scalar(
                    tensorboard_writer, 'test_loss', test_loss, epoch)
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
                print(
                    f"Epoch {epoch}: Training loss: {train_loss}. Testing loss: {test_loss}")
            if lr_scheduler is not None:
                lr_scheduler.step()
        if config.get("tensorboard", None):
            tensorboard_writer.flush()
        cls.on_train_end(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader,
                         testloader=testloader, lr_scheduler=lr_scheduler, config=config, **kwargs)
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
        pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                    disable=(kwargs.get("verbose", 1) == 0))
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
                f"epoch {epoch_idx + 1} iter {batch_idx}: train loss {loss.item():.5f}, lr:{get_lr(optimizer)}")
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
    def overfit_small_batch_(cls, net, criterion, dataset, optimizer, config=None, **kwargs):
        """This is a helper function to check if your model is working by checking if it can overfit a small dataset.

        Note:
            It uses .train_epoch().

            This function will affect the model weights and all other training-related setting/parameters.

        """
        if not isinstance(dataset, torch.utils.data.Dataset):
            warnings.warn(
                f"Currently only support Dataset as input. Got {type(dataset)}.  It will treat x as ``Dataset``")
        dataset = torch.utils.data.Subset(dataset, [0, 1, 2, 3])
        loader = DataLoader(dataset, 2)
        loss_history = []
        for epoch in range(100):
            train_loss = cls.train_epoch(net=net, criterion=criterion, trainloader=loader,
                                         optimizer=optimizer, epoch_idx=epoch, device=config['device'],
                                         config=config, **kwargs)
            loss_history.append(train_loss)
        print(loss_history)
        # del net_test
        try:
            last_loss = loss_history[-1]
            if last_loss < 1e-5:
                print(
                    "It looks like your model is working.")
        except Exception:
            pass
        print("Please check the loss_history to see whether it is overfitting. Expected to be overfit.")
        warnings.warn("Please manually re-init this model.")

    @classmethod
    def add_tensorboard_scalar(cls, writer, tag, data, step, *args, **kwargs):
        """One line code for adding data to tensorboard.
        
        Args:
            writer (SummaryWriter): the writer object.
              Put SummaryWriter Object to this argument.
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

    @classmethod
    def on_train_begin(cls, net, criterion, optimizer, trainloader, testloader=None,
                       lr_scheduler=None, config=None, **kwargs):
        """You can override this function to do something before the training.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

    @classmethod
    def on_train_end(cls, net, criterion, optimizer, trainloader, testloader=None,
                     lr_scheduler=None, config=None, **kwargs):
        """You can override this function to do something after training.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

    @classmethod
    def on_train_epoch_begin(cls, net, criterion, optimizer, trainloader, testloader=None, epoch_idx=0,
                             lr_scheduler=None, config=None, **kwargs):
        """You can override this function to do something before each epoch.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

    @classmethod
    def on_train_epoch_end(cls, net, criterion, optimizer, trainloader, testloader=None, epoch_idx=0,
                           lr_scheduler=None, config=None, **kwargs):
        """You can override this function to do something after each epoch.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

    @classmethod
    def on_test_epoch_begin(cls, net, criterion, optimizer, trainloader, testloader=None, epoch_idx=0,
                            lr_scheduler=None, config=None, **kwargs):
        """You can override this function to do something before each epoch.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

    @classmethod
    def on_test_epoch_end(cls, net, criterion, optimizer, trainloader, testloader=None, epoch_idx=0,
                          lr_scheduler=None, config=None, **kwargs):
        """You can override this function to do something after each epoch.
        Note that this does not return things. Use ``config`` to return by reference if needed.

        Args:
            net (nn.Module): this is equivalent to ``self`` or ``forward()``. Use to access instance variables.
        """
        pass

    def cuda(self, device=None):
        """set model to GPU mode, will edit ``config['device']`` to 'cuda'"""
        self._config['device'] = 'cuda'
        return super().cuda(device)

    def set_gpu(self):
        """set model to GPU mode, will edit ``config['device']`` to 'cuda'"""
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available but you are setting the model to GPU mode. This will change the "
                          "._config['device'] to cuda even though you might recieve an Exception")
        self._config['device'] = 'cuda'
        self.to('cuda')

    def cpu(self):
        """set model to CPU mode, will edit ``config['device']`` to 'cpu'"""
        self._config['device'] = 'cpu'
        return super().cpu()

    def set_cpu(self):
        """set model to CPU mode, will edit ``config['device']`` to 'cpu'"""
        self.cpu()

    def auto_gpu(self, parallel='auto', on=None):
        device, _ = btorch.utils.trainer.auto_gpu(self, parallel, on)
        self._config['device'] = device

    @property
    def device(self):
        """get the device of the model"""
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        raise Exception("You should not set device in this way. Use `model.to()` insteads.")

    def save(self, filepath, include_optimizer=True, include_lr_scheduler=True, extra=None):
        """Saves the model.state_dict and self._history.

        Args:
            filepath (str): PATH
            extra (dict): extra information to save.
        """
        to_save_optim = self._optimizer if include_optimizer else None
        to_save_lrs = self._lr_scheduler if include_lr_scheduler else None
        if extra is None:
            extra = {}
        extra['_history'] = self.history
        save_model(self, filepath, extra=extra,
                   optimizer=to_save_optim, lr_scheduler=to_save_lrs)

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

    def get_train_dataloader(self):
        """get the last train dataloader from ``.fit()``"""
        return self.train_dataloader

    def get_eval_dataloader(self):
        """get the last eval dataloader from ``.fit()``"""
        return self.eval_dataloader

    def get_test_dataloader(self):
        """get the last test dataloader from ``.evaluate()``"""
        return self.test_dataloader

    def get_predict_dataloader(self):
        """get the last predict dataloader from ``.predict()``"""
        return self.predict_dataloader


def from_pytorch(model):
    """Convert PyTroch model to BTorch model
    
    Args:
        model(torch.nn.Module): pytorch model
        
    Example:
        >>> from btorch.nn import from_pytorch
        >>> model = torch.nn.Linear(10, 10)
        >>> model = from_pytorch.model(model)
        >>> model.summary()
    """

    class bmodel(Module):
        def __init__(self, model):
            super(bmodel, self).__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    return bmodel(model)
