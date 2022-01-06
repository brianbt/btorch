import copy
from enum import auto
import warnings
from re import L
from tqdm import tqdm
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
    This class provides some highlevel training method like Keras:
        - .fit()
        - .train_net()
        - .train_epoch()
        - .test_epoch()
    You can override these class for your own usage. Just put your training loop here.

    If you decided to use the highlevel training loop. Please set 
        - self._lossfn (default:pytorch Loss Func)[required]
        - self._optimizer (default:pytorch Optimizer)[required]
        - self._lr_scheduler (default:pytorch lr_Scheduler)[optional]
        - self._config (default:dict)[optional]. For defaults usage, tt accepts:
            start_epoch = start_epoch
            max_epoch = max_epoch
            device = either cuda or cpu or auto
            save: save model path
            resume: resume model path. Override start_epoch
            val_freq = freq of running validation
    You can set them to be a pytorch instance or a dictionary (for advanced uses)
    The default guideline is only for the default highlevel functions.

    All the classmethods in this class can be taken out and use them alone. 
    They are a good starter code for traditional PyTorch training.
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

    def fit(self, x=None, y=None, batch_size=8, epochs=10, shuffle=True, drop_last=False,
            validation_split=0.0, validation_data=None, validation_batch_size=8, validation_freq=1,
            initial_epoch=None, workers=1):
        """Keras like fit method. All arguments follows Keras usage.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        It uses .train_net()

        Args:
            X: Input data. It could be:
              - torch.tensor
              - a `torch.utils.data.Dataset` dataset. Should return a tuple of `(inputs, targets)`
              - a `torch.utils.data.Dataloader`. All other dataset related argument will be ignored.
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
        if self._lossfn is None or self._optimizer is None:
            raise ValueError("`self._lossfn` and `self._optimizer` is not set.")

        self._config['max_epoch'] = epochs if epochs is not None else 10
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


        self.train_net(net=self, criterion=self._lossfn, optimizer=self._optimizer,
                   trainloader=train_loader, testloader=eval_loader, lr_scheduler=self._lr_scheduler, config=self._config)

    @classmethod
    def train_net(cls, net, criterion, optimizer, trainloader, testloader=None, lr_scheduler=None, config=None):
        """Standard PyTorch training loop. Override this function when necessary.
        It uses .train_epoch() and .test_epoch()
        
        Args:
            net ([type]): [description]
            criterion ([type]): [description]
            optimizer ([type]): [description]
            trainloader ([type]): [description]
            testloader ([type], optional): [description]. Defaults to None.
            lr_scheduler ([type], optional): [description]. Defaults to None.
            config ([type], optional): [description]. Defaults to None.
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
            train_loss = cls.train_epoch(
                net, criterion, trainloader, optimizer, epoch, device)
            train_loss_data.append(train_loss)
            if testloader is not None and epoch%val_freq==0:
                test_loss = cls.test_epoch(net, criterion, testloader, device)
                test_loss_data.append(test_loss)
            if save_path is not None:
                to_save = dict(train_loss_data=train_loss_data,
                            test_loss_data=test_loss_data)
                if test_loss <= min(test_loss_data, default=999):
                    save_model(net, f"{save_path}_best.pt",
                            to_save, optimizer, lr_scheduler)
                save_model(net, f"{save_path}_latest.pt",
                        to_save, optimizer, lr_scheduler)
                if epoch % 20 == 0:
                    save_model(net, f"{save_path}_{epoch}.pt",
                            to_save, optimizer, lr_scheduler)
            if lr_scheduler is not None:
                lr_scheduler.step()

    @classmethod
    def train_epoch(cls, net, criterion, trainloader, optimizer, epoch_idx, device='cuda', config=None):
        """This is the very basic training function for one epoch. Override this function when necessary
            
        Returns:
            (float): train_loss
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
    def test_epoch(cls, net, criterion, testloader, device='cuda'):
        """This is the very basic evaluating function for one epoch. Override this function when necessary
            
        Returns:
            (float): eval_loss
        """

        net.eval()
        train_loss = 0
        with torch.inference_mode():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                predicted = net(inputs)
                loss = criterion(predicted, targets)
                train_loss += loss.item()
        return train_loss/(batch_idx+1)

    @classmethod
    def overfit_small_batch(cls, net, criterion, dataset, optimizer):
        """This is a helper function to check if your model is working by checking if it can overfit a small dataset.
        It uses .train_epoch().
        """
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise ValueError("Currently only support Dataset as input")
        net_test = copy.deepcopy(net)
        dataset = torch.utils.data.Subset(dataset, [0,1,2,3,4])
        loader = DataLoader(dataset,5)
        loss_history = []
        for epoch in range(100):
            train_loss = cls.train_epoch(net_test, criterion, loader, optimizer, epoch)
            loss_history.append(train_loss)
        print(loss_history)
        del net_test
        try:
            last_loss = loss_history[-1].item()
            if last_loss < 1e-5:
                print("It looks like your model is working. Please double check the loss_history to see whether it is overfitting. Expected to be overfit.")
        except:
            pass
        print("Please check the loss_history to see whether it is overfitting. Expected to be overfit.")
            
    def set_gpu(self):
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available but you are setting the model to GPU mode.")
        self._config['device'] = 'cuda'
        self.to('cuda')

    def set_cpu(self):
        self._config['device'] = 'cpu'
        self.to('cpu')
        
    def auto_gpu(self, parallel='auto', on=None):
        device, _ = btorch.utils.trainer.auto_gpu(self, parallel, on)
        self._config['device'] = device
        
    # @property
    # def _lossfn(self):
    #     return self.__lossfn

    # @_lossfn.setter
    # def _lossfn(self, __lossfn):
    #     self.__lossfn = __lossfn

    # @property
    # def _optimizer(self):
    #     return self.__optimizer

    # @_optimizer.setter
    # def _optimizer(self, __optimizer):
    #     self.__optimizer = __optimizer

    # @property
    # def _lr_scheduler(self):
    #     return self.__lr_scheduler

    # @_lr_scheduler.setter
    # def _lr_scheduler(self, __lr_scheduler):
    #     self.__lr_scheduler = __lr_scheduler

    # @property
    # def _config(self):
    #     return self.__config

    # @_config.setter
    # def _config(self, __config):
    #     self.__config = __config
